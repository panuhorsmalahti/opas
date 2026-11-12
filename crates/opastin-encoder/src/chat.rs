//! Prompt formatting for the encoder model.
//!
//! Qwen3.5 uses ChatML. The text produced here matches the model's Hugging Face chat
//! template (`apply_chat_template(..., add_generation_prompt=True)`) for text-only
//! conversations; see the golden tests below. Tool definitions are not rendered yet.

use opastin::RouteRequest;

/// Whether the generation prompt opens a reasoning block.
///
/// `Thinking` ends the prompt at `<think>\n`, the point where the model would start
/// reasoning. `NonThinking` closes an empty block. The two give different final tokens,
/// so the same setting must be used for probe training and serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingMode {
    #[default]
    Thinking,
    NonThinking,
}

/// Render a request as the prompt the model would see right before answering.
pub fn format_request(request: &RouteRequest, thinking: ThinkingMode) -> String {
    let mut out = String::new();
    if let Some(system) = &request.system {
        push_turn(&mut out, "system", system);
    }
    for m in &request.messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            "system" | "developer" => "system",
            // Tool results and unknown roles are presented as user turns.
            _ => "user",
        };
        push_turn(&mut out, role, &m.text);
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(match thinking {
        ThinkingMode::Thinking => "<think>\n",
        ThinkingMode::NonThinking => "<think>\n\n</think>\n\n",
    });
    out
}

/// A single user prompt.
pub fn format_prompt(prompt: &str, thinking: ThinkingMode) -> String {
    let request = RouteRequest {
        format: opastin::ApiFormat::OpenaiChat,
        model: None,
        system: None,
        messages: vec![opastin::Message {
            role: "user".into(),
            text: prompt.into(),
        }],
        has_tools: false,
        stream: false,
        continuation_id: None,
        conversation: None,
    };
    format_request(&request, thinking)
}

fn push_turn(out: &mut String, role: &str, content: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(content);
    out.push_str("<|im_end|>\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use opastin::ApiFormat;
    use serde_json::json;

    // Expected strings come from the Qwen3.5 HF chat template (transformers
    // apply_chat_template, add_generation_prompt=True, enable_thinking unset).
    const NON_THINKING_SUFFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    #[test]
    fn matches_hf_template_user_only() {
        assert_eq!(
            format_prompt("What is 2 + 2?", ThinkingMode::NonThinking),
            format!("<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n{NON_THINKING_SUFFIX}")
        );
    }

    #[test]
    fn matches_hf_template_multi_turn_with_system() {
        let body = json!({"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Another?"}
        ]});
        let req = RouteRequest::from_json(ApiFormat::OpenaiChat, &body).unwrap();
        assert_eq!(
            format_request(&req, ThinkingMode::NonThinking),
            format!(
                "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nName a prime.<|im_end|>\n\
                 <|im_start|>assistant\n7<|im_end|>\n<|im_start|>user\nAnother?<|im_end|>\n{NON_THINKING_SUFFIX}"
            )
        );
    }

    #[test]
    fn matches_hf_template_anthropic_system_and_multiline() {
        let body = json!({
            "system": "You are terse.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "line one\nline two\n\n  indented"}]}]
        });
        let req = RouteRequest::from_json(ApiFormat::AnthropicMessages, &body).unwrap();
        assert_eq!(
            format_request(&req, ThinkingMode::NonThinking),
            format!(
                "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nline one\nline two\n\n  indented<|im_end|>\n{NON_THINKING_SUFFIX}"
            )
        );
    }

    #[test]
    fn thinking_mode_ends_at_open_think_block() {
        assert!(format_prompt("hi", ThinkingMode::Thinking)
            .ends_with("<|im_start|>assistant\n<think>\n"));
    }
}
