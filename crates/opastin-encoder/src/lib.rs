//! In-process prefill encoder: runs a GGUF model with llama.cpp and captures the hidden
//! state after every decoder layer, without generating any tokens.
//!
//! One `llama_decode` call processes the prompt (the prefill). llama.cpp's graph-eval
//! callback (`cb_eval`) is asked to expose the tensors named `model.input_embed` (the
//! embedding output, index 0) and `l_out-<i>` (the residual stream after decoder layer
//! `i`, index `i + 1`). Each is copied off the compute device as it is produced and
//! pooled two ways: the last prompt token, and the mean over all prompt tokens.
//!
//! ```no_run
//! use opastin_encoder::{Encoder, EncoderConfig};
//!
//! let mut enc = Encoder::load("models/gguf/Qwen3.5-9B-Q8_0.gguf", EncoderConfig::default())?;
//! let acts = enc.encode_prompt("Fix the race condition in this worker pool.")?;
//! println!("{} tokens, {} layers x {} dims", acts.n_tokens, acts.layers.len(), enc.n_embd());
//! # Ok::<(), opastin_encoder::EncoderError>(())
//! ```

pub mod chat;
pub mod npy;
pub mod probe;
pub mod router;

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::time::{Duration, Instant};

use llama_cpp_sys_2 as sys;
use opastin::RouteRequest;

pub use chat::{format_prompt, format_request, ThinkingMode};
pub use probe::{Probe, ProbeFile};
pub use router::ProbeRouter;

#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("invalid model path: {0}")]
    BadPath(String),
    #[error("failed to load model from {0}")]
    Load(String),
    #[error("failed to create llama.cpp context")]
    Context,
    #[error("layer index {index} out of range (model has {n_layer} layers; valid indices are 0..={n_layer})")]
    BadLayer { index: usize, n_layer: usize },
    #[error("tokenization failed")]
    Tokenize,
    #[error("prompt is empty")]
    Empty,
    #[error("llama_decode failed with code {0}")]
    Decode(i32),
    #[error("activation capture failed: {0}")]
    Capture(String),
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Layers to offload to the GPU; negative means all.
    pub n_gpu_layers: i32,
    /// Prompts longer than this keep their first and last `max_tokens / 2` tokens: the
    /// start holds the system prompt and task statement, the end the latest turn and the
    /// open assistant turn where the features are read.
    pub max_tokens: usize,
    /// Hidden-state indices to capture (0 = embeddings, i = after layer i). `None` = all.
    pub layers: Option<Vec<usize>>,
    pub thinking: ThinkingMode,
    /// Print llama.cpp's log output (errors are always printed).
    pub verbose: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            max_tokens: 8192,
            layers: None,
            thinking: ThinkingMode::Thinking,
            verbose: false,
        }
    }
}

/// Pooled hidden states for one prompt.
#[derive(Debug, Clone)]
pub struct Activations {
    /// Prompt length in tokens after truncation.
    pub n_tokens: usize,
    /// Captured hidden-state indices, ascending (0 = embeddings, i = after layer i).
    pub layers: Vec<usize>,
    /// `last[k]` is the state of the final prompt token at `layers[k]`.
    pub last: Vec<Vec<f32>>,
    /// `mean[k]` is the mean over all prompt tokens at `layers[k]`.
    pub mean: Vec<Vec<f32>>,
    /// Wall time of the prefill and capture.
    pub elapsed: Duration,
}

/// State shared with the eval callback. Boxed so its address is stable.
struct Capture {
    n_embd: usize,
    n_layer: usize,
    wanted: Vec<bool>,
    sum: Vec<Vec<f64>>,
    last: Vec<Vec<f32>>,
    seen: Vec<usize>,
    buf: Vec<f32>,
    error: Option<String>,
}

impl Capture {
    fn new(n_embd: usize, n_layer: usize, layers: &[usize]) -> Self {
        let mut wanted = vec![false; n_layer + 1];
        for &l in layers {
            wanted[l] = true;
        }
        let alloc = |w: &bool| if *w { n_embd } else { 0 };
        Self {
            n_embd,
            n_layer,
            sum: wanted.iter().map(|w| vec![0.0; alloc(w)]).collect(),
            last: wanted.iter().map(|w| vec![0.0; alloc(w)]).collect(),
            seen: vec![0; n_layer + 1],
            wanted,
            buf: Vec::new(),
            error: None,
        }
    }

    fn reset(&mut self) {
        self.sum.iter_mut().for_each(|s| s.fill(0.0));
        self.seen.fill(0);
        self.error = None;
    }

    /// Map a graph tensor name to a hidden-state index.
    fn index_of(&self, name: &CStr) -> Option<usize> {
        let name = name.to_str().ok()?;
        if name == "model.input_embed" {
            return Some(0);
        }
        let layer: usize = name.strip_prefix("l_out-")?.parse().ok()?;
        (layer < self.n_layer).then_some(layer + 1)
    }

    /// Copy one captured tensor (shape [n_embd, n_tokens_in_ubatch]) and accumulate.
    unsafe fn absorb(&mut self, index: usize, t: *mut sys::ggml_tensor) {
        let tensor = &*t;
        if tensor.type_ != sys::GGML_TYPE_F32 {
            self.error = Some(format!("tensor for index {index} is not f32"));
            return;
        }
        if !sys::ggml_is_contiguous(t) {
            self.error = Some(format!("tensor for index {index} is not contiguous"));
            return;
        }
        let ne0 = tensor.ne[0] as usize;
        let rows = (tensor.ne[1] * tensor.ne[2] * tensor.ne[3]) as usize;
        if ne0 != self.n_embd || rows == 0 {
            self.error = Some(format!(
                "tensor for index {index} has shape [{ne0}, {rows}], expected [{}, n]",
                self.n_embd
            ));
            return;
        }
        self.buf.resize(ne0 * rows, 0.0);
        sys::ggml_backend_tensor_get(t, self.buf.as_mut_ptr().cast(), 0, sys::ggml_nbytes(t));

        let sum = &mut self.sum[index];
        for row in self.buf.chunks_exact(ne0) {
            for (s, v) in sum.iter_mut().zip(row) {
                *s += f64::from(*v);
            }
        }
        // Ubatches arrive in prompt order, so the last row seen is the final token.
        self.last[index].copy_from_slice(&self.buf[(rows - 1) * ne0..]);
        self.seen[index] += rows;
    }
}

/// llama.cpp graph-eval callback. With `ask == true` the scheduler asks whether we want
/// to see a tensor; with `ask == false` the tensor has been computed and can be read.
unsafe extern "C" fn eval_callback(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    let capture = &mut *(user_data as *mut Capture);
    let name = CStr::from_ptr((*t).name.as_ptr());
    let Some(index) = capture.index_of(name).filter(|&i| capture.wanted[i]) else {
        // Not ours: don't observe it; if asked after the fact, keep computing.
        return !ask;
    };
    if !ask {
        capture.absorb(index, t);
    }
    true
}

static VERBOSE: AtomicBool = AtomicBool::new(false);
static BACKEND_INIT: Once = Once::new();

unsafe extern "C" fn log_callback(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    if VERBOSE.load(Ordering::Relaxed) || level >= sys::GGML_LOG_LEVEL_ERROR {
        eprint!("{}", CStr::from_ptr(text).to_string_lossy());
    }
}

/// Replaces llama.cpp's default progress dots during model loading.
unsafe extern "C" fn silent_progress(_progress: f32, _: *mut c_void) -> bool {
    true
}

/// A loaded model plus one llama.cpp context. Not `Sync`: wrap in a `Mutex` to share.
pub struct Encoder {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    capture: Box<Capture>,
    config: EncoderConfig,
    layers: Vec<usize>,
}

// The raw pointers are owned exclusively by this struct; llama.cpp contexts may be used
// from any one thread at a time.
unsafe impl Send for Encoder {}

impl Encoder {
    pub fn load(path: impl AsRef<Path>, config: EncoderConfig) -> Result<Self, EncoderError> {
        let path = path.as_ref();
        let cpath = path
            .to_str()
            .and_then(|s| CString::new(s).ok())
            .ok_or_else(|| EncoderError::BadPath(path.display().to_string()))?;

        VERBOSE.store(config.verbose, Ordering::Relaxed);
        BACKEND_INIT.call_once(|| unsafe {
            sys::llama_log_set(Some(log_callback), std::ptr::null_mut());
            sys::llama_backend_init();
        });

        unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.n_gpu_layers = config.n_gpu_layers;
            mp.load_mtp = false;
            if !config.verbose {
                mp.progress_callback = Some(silent_progress);
            }
            let model = sys::llama_model_load_from_file(cpath.as_ptr(), mp);
            if model.is_null() {
                return Err(EncoderError::Load(path.display().to_string()));
            }

            let n_embd = sys::llama_model_n_embd(model) as usize;
            let n_layer = sys::llama_model_n_layer(model) as usize;
            let layers: Vec<usize> = match &config.layers {
                Some(l) => {
                    let mut l = l.clone();
                    l.sort_unstable();
                    l.dedup();
                    if let Some(&bad) = l.iter().find(|&&i| i > n_layer) {
                        sys::llama_model_free(model);
                        return Err(EncoderError::BadLayer {
                            index: bad,
                            n_layer,
                        });
                    }
                    l
                }
                None => (0..=n_layer).collect(),
            };
            let mut capture = Box::new(Capture::new(n_embd, n_layer, &layers));

            let mut cp = sys::llama_context_default_params();
            let n_ctx = config.max_tokens.max(16) as u32;
            cp.n_ctx = n_ctx;
            cp.n_batch = n_ctx;
            cp.n_ubatch = n_ctx.min(512);
            cp.n_seq_max = 1;
            cp.embeddings = false;
            cp.no_perf = true;
            cp.cb_eval = Some(eval_callback);
            cp.cb_eval_user_data = (&mut *capture as *mut Capture).cast();
            let ctx = sys::llama_init_from_model(model, cp);
            if ctx.is_null() {
                sys::llama_model_free(model);
                return Err(EncoderError::Context);
            }

            Ok(Self {
                model,
                ctx,
                vocab: sys::llama_model_get_vocab(model),
                capture,
                config,
                layers,
            })
        }
    }

    pub fn n_embd(&self) -> usize {
        self.capture.n_embd
    }

    /// Number of decoder layers (hidden-state indices run 0..=n_layer).
    pub fn n_layer(&self) -> usize {
        self.capture.n_layer
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Encode an OpenAI or Anthropic request as the model would see it before answering.
    pub fn encode_request(&mut self, request: &RouteRequest) -> Result<Activations, EncoderError> {
        let text = format_request(request, self.config.thinking);
        self.encode_formatted(&text)
    }

    /// Encode a single user prompt.
    pub fn encode_prompt(&mut self, prompt: &str) -> Result<Activations, EncoderError> {
        let text = format_prompt(prompt, self.config.thinking);
        self.encode_formatted(&text)
    }

    /// Encode already-formatted prompt text (special tokens such as `<|im_start|>` are parsed).
    pub fn encode_formatted(&mut self, text: &str) -> Result<Activations, EncoderError> {
        let mut tokens = self.tokenize(text)?;
        if tokens.is_empty() {
            return Err(EncoderError::Empty);
        }
        truncate_middle(&mut tokens, self.config.max_tokens);
        let n = tokens.len();

        self.capture.reset();
        let start = Instant::now();
        let rc = unsafe {
            sys::llama_memory_clear(sys::llama_get_memory(self.ctx), true);
            let batch = sys::llama_batch_get_one(tokens.as_mut_ptr(), n as i32);
            let rc = sys::llama_decode(self.ctx, batch);
            sys::llama_synchronize(self.ctx);
            rc
        };
        let elapsed = start.elapsed();
        if rc != 0 {
            return Err(EncoderError::Decode(rc));
        }
        if let Some(e) = self.capture.error.take() {
            return Err(EncoderError::Capture(e));
        }
        for &i in &self.layers {
            if self.capture.seen[i] != n {
                return Err(EncoderError::Capture(format!(
                    "index {i}: captured {} of {n} tokens",
                    self.capture.seen[i]
                )));
            }
        }

        let mean = self
            .layers
            .iter()
            .map(|&i| {
                self.capture.sum[i]
                    .iter()
                    .map(|s| (s / n as f64) as f32)
                    .collect()
            })
            .collect();
        let last = self
            .layers
            .iter()
            .map(|&i| self.capture.last[i].clone())
            .collect();
        Ok(Activations {
            n_tokens: n,
            layers: self.layers.clone(),
            last,
            mean,
            elapsed,
        })
    }

    /// Token ids for formatted text, without adding BOS (matching the HF chat template).
    pub fn tokenize(&self, text: &str) -> Result<Vec<sys::llama_token>, EncoderError> {
        let len = i32::try_from(text.len()).map_err(|_| EncoderError::Tokenize)?;
        let mut tokens: Vec<sys::llama_token> = vec![0; text.len() + 16];
        for _ in 0..2 {
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast(),
                    len,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    false,
                    true,
                )
            };
            if n >= 0 {
                tokens.truncate(n as usize);
                return Ok(tokens);
            }
            if n == i32::MIN {
                break;
            }
            tokens.resize((-n) as usize, 0);
        }
        Err(EncoderError::Tokenize)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

/// Drop tokens from the middle so that at most `max` remain: the first `max / 2` and
/// the last `max - max / 2` are kept.
pub fn truncate_middle<T>(tokens: &mut Vec<T>, max: usize) {
    if tokens.len() > max {
        let head = max / 2;
        let tail = max - head;
        tokens.drain(head..tokens.len() - tail);
    }
}

/// Cosine similarity of two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)) as f32
}

#[cfg(test)]
mod tests {
    use super::truncate_middle;

    #[test]
    fn truncation_keeps_both_ends() {
        let mut v: Vec<u32> = (0..10).collect();
        truncate_middle(&mut v, 5);
        assert_eq!(v, [0, 1, 7, 8, 9]);
        let mut v: Vec<u32> = (0..4).collect();
        truncate_middle(&mut v, 5);
        assert_eq!(v, [0, 1, 2, 3]);
    }
}
