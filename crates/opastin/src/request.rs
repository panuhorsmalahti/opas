//! A format-independent view of an LLM request, for routers to inspect.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::ApiFormat;
use crate::conversation::Conversation;

/// One message, reduced to its role and text content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub text: String,
}

/// What a router sees of a request. Built from an OpenAI or Anthropic request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteRequest {
    pub format: ApiFormat,
    /// The `model` field the client sent, if any.
    pub model: Option<String>,
    /// System prompt text (OpenAI `system`/`developer` messages, Anthropic `system`).
    pub system: Option<String>,
    /// Non-system messages in order.
    pub messages: Vec<Message>,
    /// Whether the request declares tools.
    pub has_tools: bool,
    /// Whether the client asked for a streamed response.
    pub stream: bool,
    /// The conversation this request continues, if known (filled in by the gateway for
    /// cache-aware routing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Conversation>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RequestError {
    #[error("request body must be a JSON object")]
    NotAnObject,
    #[error("request body is missing a `messages` array")]
    MissingMessages,
}

impl RouteRequest {
    /// Parse a request body in the given format.
    pub fn from_json(format: ApiFormat, body: &Value) -> Result<Self, RequestError> {
        match format {
            ApiFormat::OpenaiChat => Self::from_openai(body),
            ApiFormat::AnthropicMessages => Self::from_anthropic(body),
        }
    }

    /// Parse an OpenAI Chat Completions request body.
    pub fn from_openai(body: &Value) -> Result<Self, RequestError> {
        let obj = body.as_object().ok_or(RequestError::NotAnObject)?;
        let raw = obj
            .get("messages")
            .and_then(Value::as_array)
            .ok_or(RequestError::MissingMessages)?;

        let mut system = Vec::new();
        let mut messages = Vec::new();
        for m in raw {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let text = content_text(m.get("content"));
            if role == "system" || role == "developer" {
                system.push(text);
            } else {
                messages.push(Message {
                    role: role.to_owned(),
                    text,
                });
            }
        }

        Ok(Self {
            format: ApiFormat::OpenaiChat,
            model: str_field(body, "model"),
            system: join_nonempty(system),
            messages,
            has_tools: nonempty_array(body, "tools") || nonempty_array(body, "functions"),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            conversation: None,
        })
    }

    /// Parse an Anthropic Messages request body.
    pub fn from_anthropic(body: &Value) -> Result<Self, RequestError> {
        let obj = body.as_object().ok_or(RequestError::NotAnObject)?;
        let raw = obj
            .get("messages")
            .and_then(Value::as_array)
            .ok_or(RequestError::MissingMessages)?;

        let mut system = Vec::new();
        let mut messages = Vec::new();
        for m in raw {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let text = content_text(m.get("content"));
            // Mid-conversation system messages are allowed in the Messages API.
            if role == "system" {
                system.push(text);
            } else {
                messages.push(Message {
                    role: role.to_owned(),
                    text,
                });
            }
        }
        if let Some(top) = obj.get("system") {
            system.insert(0, content_text(Some(top)));
        }

        Ok(Self {
            format: ApiFormat::AnthropicMessages,
            model: str_field(body, "model"),
            system: join_nonempty(system),
            messages,
            has_tools: nonempty_array(body, "tools"),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            conversation: None,
        })
    }

    /// Text of the most recent user message, if any.
    pub fn last_user_text(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.text.as_str())
    }
}

/// Extract text from a content field: a plain string, or an array of parts/blocks
/// where `{"type": "text", "text": ...}` entries are kept (tool results with string or
/// nested text content are included too).
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).map(str::to_owned),
                Some("tool_result") => Some(content_text(p.get("content"))),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn str_field(body: &Value, key: &str) -> Option<String> {
    body.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn nonempty_array(body: &Value, key: &str) -> bool {
    body.get(key)
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

fn join_nonempty(parts: Vec<String>) -> Option<String> {
    let parts: Vec<_> = parts.into_iter().filter(|s| !s.is_empty()).collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_openai_request() {
        let body = json!({
            "model": "auto",
            "stream": true,
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "fix my bug"}
            ],
            "tools": [{"type": "function", "function": {"name": "f"}}]
        });
        let r = RouteRequest::from_openai(&body).unwrap();
        assert_eq!(r.model.as_deref(), Some("auto"));
        assert_eq!(r.system.as_deref(), Some("be brief"));
        assert_eq!(r.messages.len(), 3);
        assert_eq!(r.messages[0].text, "hi");
        assert_eq!(r.last_user_text(), Some("fix my bug"));
        assert!(r.has_tools);
        assert!(r.stream);
    }

    #[test]
    fn parses_anthropic_request() {
        let body = json!({
            "model": "auto",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "you are a coder"}],
            "messages": [
                {"role": "user", "content": "write a test"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
            ]
        });
        let r = RouteRequest::from_anthropic(&body).unwrap();
        assert_eq!(r.system.as_deref(), Some("you are a coder"));
        assert_eq!(r.messages.len(), 3);
        assert_eq!(r.messages[1].text, "");
        assert_eq!(r.last_user_text(), Some("ok"));
        assert!(!r.has_tools);
        assert!(!r.stream);
    }

    #[test]
    fn rejects_bad_bodies() {
        assert_eq!(
            RouteRequest::from_openai(&json!([1, 2])),
            Err(RequestError::NotAnObject)
        );
        assert_eq!(
            RouteRequest::from_anthropic(&json!({"model": "x"})),
            Err(RequestError::MissingMessages)
        );
    }
}
