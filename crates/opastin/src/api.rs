use serde::{Deserialize, Serialize};

/// The wire format of an LLM request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiFormat {
    /// OpenAI Chat Completions (`POST /v1/chat/completions`).
    OpenaiChat,
    /// Anthropic Messages (`POST /v1/messages`).
    AnthropicMessages,
    /// Open Responses, the open specification of OpenAI's Responses API
    /// (`POST /v1/responses`, <https://www.openresponses.org>).
    OpenResponses,
    /// Google Gemini Interactions API (`POST /v1beta/interactions` or `/v1/interactions`).
    GeminiInteractions,
}

impl ApiFormat {
    pub const ALL: [ApiFormat; 4] = [
        ApiFormat::OpenaiChat,
        ApiFormat::AnthropicMessages,
        ApiFormat::OpenResponses,
        ApiFormat::GeminiInteractions,
    ];

    /// Path appended to an upstream base URL (which ends in the API version, e.g. `/v1`).
    pub fn endpoint_path(self) -> &'static str {
        match self {
            ApiFormat::OpenaiChat => "/chat/completions",
            ApiFormat::AnthropicMessages => "/messages",
            ApiFormat::OpenResponses => "/responses",
            ApiFormat::GeminiInteractions => "/interactions",
        }
    }

    /// Request field that continues a conversation stored at the provider, for APIs
    /// that keep server-side state.
    pub fn continuation_field(self) -> Option<&'static str> {
        match self {
            ApiFormat::OpenResponses => Some("previous_response_id"),
            ApiFormat::GeminiInteractions => Some("previous_interaction_id"),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ApiFormat::OpenaiChat => "openai_chat",
            ApiFormat::AnthropicMessages => "anthropic_messages",
            ApiFormat::OpenResponses => "open_responses",
            ApiFormat::GeminiInteractions => "gemini_interactions",
        }
    }
}

impl std::fmt::Display for ApiFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
