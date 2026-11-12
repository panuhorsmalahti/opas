use serde::{Deserialize, Serialize};

/// The wire format of an LLM request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiFormat {
    /// OpenAI Chat Completions (`POST /v1/chat/completions`).
    OpenaiChat,
    /// Anthropic Messages (`POST /v1/messages`).
    AnthropicMessages,
}

impl ApiFormat {
    /// Path appended to an upstream base URL (which ends in `/v1`).
    pub fn endpoint_path(self) -> &'static str {
        match self {
            ApiFormat::OpenaiChat => "/chat/completions",
            ApiFormat::AnthropicMessages => "/messages",
        }
    }
}

impl std::fmt::Display for ApiFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ApiFormat::OpenaiChat => "openai_chat",
            ApiFormat::AnthropicMessages => "anthropic_messages",
        })
    }
}
