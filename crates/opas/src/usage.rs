//! Token usage reported by providers, read from responses in any supported format, and
//! its cost.
//!
//! Each API reports usage in its own fields and events:
//!
//! | API | Where | Fields |
//! |---|---|---|
//! | Anthropic Messages | body, or `message_start` + `message_delta` events | `input_tokens` (uncached only), `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens` |
//! | OpenAI Chat Completions | body, or the last chunk when `stream_options.include_usage` is set | `prompt_tokens`, `prompt_tokens_details.cached_tokens`, `completion_tokens` |
//! | Open Responses | body, or the `response.completed` event | `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens` |
//! | Gemini Interactions | body, or the final event | `total_input_tokens`, `total_cached_tokens`, `total_output_tokens` |
//!
//! [`UsageScanner`] watches a response body as it streams past (whole JSON bodies and
//! SSE `data:` lines alike) and keeps the latest value of each field, so cumulative
//! counts in later events win.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::conversation::Pricing;

/// Token counts for one request, normalized across APIs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    /// All prompt tokens, including those read from or written to the prompt cache.
    pub prompt_tokens: u64,
    /// Prompt tokens read from the prompt cache.
    pub cached_prompt_tokens: u64,
    /// Prompt tokens written to the prompt cache (Anthropic reports these separately).
    pub cache_write_tokens: u64,
    /// Output tokens, including reasoning / thinking tokens.
    pub output_tokens: u64,
}

impl Usage {
    /// Cost in USD at `pricing` (USD per million tokens). Cache writes are charged at the
    /// input price; providers that charge a premium for them make this a slight
    /// underestimate.
    pub fn cost_usd(&self, pricing: &Pricing) -> f64 {
        let p = pricing.for_prompt(self.prompt_tokens);
        let cached = self.cached_prompt_tokens.min(self.prompt_tokens);
        let uncached = self.prompt_tokens - cached;
        (uncached as f64 * p.input
            + cached as f64 * p.cached_input
            + self.output_tokens as f64 * p.output)
            / 1e6
    }
}

/// Raw fields seen so far; `None` until a field appears.
#[derive(Debug, Default, Clone, Copy)]
struct Seen {
    input: Option<u64>,
    cached: Option<u64>,
    cache_write: Option<u64>,
    output: Option<u64>,
}

/// Collects usage from a response body as it streams past.
#[derive(Debug)]
pub struct UsageScanner {
    format: ApiFormat,
    sse: bool,
    /// SSE: the incomplete last line. JSON: the body so far.
    buf: Vec<u8>,
    overflow: bool,
    seen: Seen,
}

/// Largest non-streamed body that is parsed for usage.
const MAX_JSON_BODY: usize = 16 * 1024 * 1024;
/// Longest SSE line that is parsed.
const MAX_SSE_LINE: usize = 4 * 1024 * 1024;

impl UsageScanner {
    /// `content_type` is the response's `content-type`; `text/event-stream` selects SSE.
    pub fn new(format: ApiFormat, content_type: Option<&str>) -> Self {
        let sse = content_type.is_some_and(|c| c.starts_with("text/event-stream"));
        Self {
            format,
            sse,
            buf: Vec::new(),
            overflow: false,
            seen: Seen::default(),
        }
    }

    /// Feed the next chunk of the body.
    pub fn feed(&mut self, chunk: &[u8]) {
        if !self.sse {
            if self.buf.len() + chunk.len() > MAX_JSON_BODY {
                self.overflow = true;
                self.buf.clear();
            } else if !self.overflow {
                self.buf.extend_from_slice(chunk);
            }
            return;
        }
        self.buf.extend_from_slice(chunk);
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            self.sse_line(&line);
        }
        if self.buf.len() > MAX_SSE_LINE {
            self.buf.clear();
        }
    }

    fn sse_line(&mut self, line: &[u8]) {
        let Ok(text) = std::str::from_utf8(line) else {
            return;
        };
        let Some(data) = text.trim_end().strip_prefix("data:") else {
            return;
        };
        if let Ok(v) = serde_json::from_str::<Value>(data.trim()) {
            self.observe(&v);
        }
    }

    fn observe(&mut self, v: &Value) {
        for usage in [
            v.get("usage"),
            v.pointer("/message/usage"),
            v.pointer("/response/usage"),
            v.pointer("/interaction/usage"),
        ]
        .into_iter()
        .flatten()
        {
            let num = |paths: &[&str]| {
                paths
                    .iter()
                    .find_map(|p| usage.pointer(p).and_then(Value::as_u64))
            };
            let s = &mut self.seen;
            if let Some(n) = num(&["/input_tokens", "/prompt_tokens", "/total_input_tokens"]) {
                s.input = Some(n);
            }
            if let Some(n) = num(&[
                "/cache_read_input_tokens",
                "/prompt_tokens_details/cached_tokens",
                "/input_tokens_details/cached_tokens",
                "/total_cached_tokens",
            ]) {
                s.cached = Some(n);
            }
            if let Some(n) = num(&["/cache_creation_input_tokens"]) {
                s.cache_write = Some(n);
            }
            if let Some(n) = num(&[
                "/output_tokens",
                "/completion_tokens",
                "/total_output_tokens",
            ]) {
                s.output = Some(n);
            }
        }
    }

    /// The usage seen, or `None` if the response reported none (e.g. an error, or an
    /// OpenAI stream without `stream_options.include_usage`).
    pub fn finish(mut self) -> Option<Usage> {
        if self.sse {
            let rest = std::mem::take(&mut self.buf);
            self.sse_line(&rest);
        } else if !self.overflow {
            if let Ok(v) = serde_json::from_slice::<Value>(&self.buf) {
                self.observe(&v);
            }
        }
        let s = self.seen;
        if s.input.is_none() && s.output.is_none() {
            return None;
        }
        let cached = s.cached.unwrap_or(0);
        let cache_write = s.cache_write.unwrap_or(0);
        let input = s.input.unwrap_or(0);
        // Anthropic's `input_tokens` excludes cache reads and writes; the other APIs'
        // prompt counts include cached tokens.
        let prompt_tokens = match self.format {
            ApiFormat::AnthropicMessages => input + cached + cache_write,
            _ => input,
        };
        Some(Usage {
            prompt_tokens,
            cached_prompt_tokens: cached,
            cache_write_tokens: cache_write,
            output_tokens: s.output.unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(format: ApiFormat, content_type: &str, chunks: &[&str]) -> Option<Usage> {
        let mut s = UsageScanner::new(format, Some(content_type));
        for c in chunks {
            s.feed(c.as_bytes());
        }
        s.finish()
    }

    #[test]
    fn anthropic_body_and_stream() {
        let body = r#"{"type":"message","usage":{"input_tokens":20,"cache_read_input_tokens":1000,"cache_creation_input_tokens":5,"output_tokens":300}}"#;
        let u = scan(
            ApiFormat::AnthropicMessages,
            "application/json",
            &[&body[..30], &body[30..]],
        )
        .unwrap();
        assert_eq!(
            u,
            Usage {
                prompt_tokens: 1025,
                cached_prompt_tokens: 1000,
                cache_write_tokens: 5,
                output_tokens: 300
            }
        );

        let sse = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"cache_read_input_tokens\":0,\"output_tokens\":1}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\nevent: message_del",
            "ta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":87}}\n\n",
        ];
        let u = scan(ApiFormat::AnthropicMessages, "text/event-stream", &sse).unwrap();
        assert_eq!(
            (u.prompt_tokens, u.output_tokens),
            (12, 87),
            "the later cumulative count wins"
        );
    }

    #[test]
    fn openai_responses_and_gemini() {
        let chat = r#"{"usage":{"prompt_tokens":1200,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":1024}}}"#;
        let u = scan(ApiFormat::OpenaiChat, "application/json", &[chat]).unwrap();
        assert_eq!(
            (u.prompt_tokens, u.cached_prompt_tokens, u.output_tokens),
            (1200, 1024, 50)
        );
        // A stream without include_usage reports nothing.
        assert!(scan(
            ApiFormat::OpenaiChat,
            "text/event-stream",
            &["data: {\"choices\":[]}\n\ndata: [DONE]\n\n"]
        )
        .is_none());

        let completed = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":40,\"input_tokens_details\":{\"cached_tokens\":8},\"output_tokens\":9}}}\n\n";
        let u = scan(ApiFormat::OpenResponses, "text/event-stream", &[completed]).unwrap();
        assert_eq!(
            (u.prompt_tokens, u.cached_prompt_tokens, u.output_tokens),
            (40, 8, 9)
        );

        let gemini = r#"{"id":"i1","usage":{"total_input_tokens":30,"total_cached_tokens":0,"total_output_tokens":70}}"#;
        let u = scan(ApiFormat::GeminiInteractions, "application/json", &[gemini]).unwrap();
        assert_eq!((u.prompt_tokens, u.output_tokens), (30, 70));
        assert!(scan(
            ApiFormat::GeminiInteractions,
            "application/json",
            &["{\"error\":{}}"]
        )
        .is_none());
    }

    #[test]
    fn cost_uses_cached_and_long_context_prices() {
        let p = Pricing {
            input: 4.0,
            cached_input: 0.4,
            output: 20.0,
            long_context: Some(crate::conversation::LongContextPricing {
                above_tokens: 100_000,
                input: 8.0,
                cached_input: 0.8,
                output: 40.0,
            }),
        };
        let u = Usage {
            prompt_tokens: 10_000,
            cached_prompt_tokens: 8_000,
            cache_write_tokens: 0,
            output_tokens: 1_000,
        };
        let want = (2_000.0 * 4.0 + 8_000.0 * 0.4 + 1_000.0 * 20.0) / 1e6;
        assert!((u.cost_usd(&p) - want).abs() < 1e-12);
        let long = Usage {
            prompt_tokens: 200_000,
            ..u
        };
        let want = (192_000.0 * 8.0 + 8_000.0 * 0.8 + 1_000.0 * 40.0) / 1e6;
        assert!((long.cost_usd(&p) - want).abs() < 1e-12);
    }
}
