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

/// What a router sees of a request. Built from an OpenAI Chat Completions, Anthropic
/// Messages, Open Responses or Gemini Interactions request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteRequest {
    pub format: ApiFormat,
    /// The `model` field the client sent, if any.
    pub model: Option<String>,
    /// System prompt text (OpenAI `system`/`developer` messages, Anthropic `system`,
    /// Open Responses `instructions`, Gemini `system_instruction`).
    pub system: Option<String>,
    /// Non-system messages in order.
    pub messages: Vec<Message>,
    /// Whether the request declares tools.
    pub has_tools: bool,
    /// Whether the client asked for a streamed response.
    pub stream: bool,
    /// Id of a provider-stored conversation this request continues
    /// (`previous_response_id` / `previous_interaction_id`). The provider holds the earlier
    /// turns, so such a request can only be served by the model that produced that id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_id: Option<String>,
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
    #[error("request body needs `input` or `{0}`")]
    MissingInput(&'static str),
}

impl RouteRequest {
    /// Parse a request body in the given format.
    pub fn from_json(format: ApiFormat, body: &Value) -> Result<Self, RequestError> {
        match format {
            ApiFormat::OpenaiChat => Self::from_openai(body),
            ApiFormat::AnthropicMessages => Self::from_anthropic(body),
            ApiFormat::OpenResponses => Self::from_open_responses(body),
            ApiFormat::GeminiInteractions => Self::from_gemini_interactions(body),
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
            continuation_id: None,
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
            continuation_id: None,
            conversation: None,
        })
    }

    /// Parse an Open Responses (OpenAI Responses API) request body.
    ///
    /// `instructions` becomes the system prompt. `input` is a string (one user message)
    /// or a list of items: messages (roles user / assistant / system / developer, content
    /// as a string or `input_text` / `output_text` / `refusal` parts), `function_call`
    /// and `function_call_output`. Reasoning, compaction and item references carry no
    /// readable text and are skipped.
    pub fn from_open_responses(body: &Value) -> Result<Self, RequestError> {
        let obj = body.as_object().ok_or(RequestError::NotAnObject)?;
        let continuation_id = str_field(body, "previous_response_id");
        let mut system = vec![];
        if let Some(Value::String(i)) = obj.get("instructions") {
            system.push(i.clone());
        }
        let mut messages = Vec::new();
        match obj.get("input") {
            Some(Value::String(text)) => messages.push(Message {
                role: "user".into(),
                text: text.clone(),
            }),
            Some(Value::Array(items)) => {
                for item in items {
                    let kind = item.get("type").and_then(Value::as_str);
                    let role = item.get("role").and_then(Value::as_str);
                    match (kind, role) {
                        (Some("message") | None, Some(role)) => {
                            let text = content_text(item.get("content"));
                            match role {
                                "system" | "developer" => system.push(text),
                                "assistant" => messages.push(Message {
                                    role: "assistant".into(),
                                    text,
                                }),
                                _ => messages.push(Message {
                                    role: "user".into(),
                                    text,
                                }),
                            }
                        }
                        (Some("function_call"), _) => messages.push(Message {
                            role: "assistant".into(),
                            text: tool_call_text(item.get("name"), item.get("arguments")),
                        }),
                        (Some("function_call_output"), _) => messages.push(Message {
                            role: "user".into(),
                            text: content_text(item.get("output")),
                        }),
                        _ => {}
                    }
                }
            }
            None | Some(Value::Null) if continuation_id.is_some() => {}
            _ => return Err(RequestError::MissingInput("previous_response_id")),
        }
        Ok(Self {
            format: ApiFormat::OpenResponses,
            model: str_field(body, "model"),
            system: join_nonempty(system),
            messages,
            has_tools: nonempty_array(body, "tools"),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            continuation_id,
            conversation: None,
        })
    }

    /// Parse a Gemini Interactions API request body.
    ///
    /// `system_instruction` becomes the system prompt. `input` is a string, a content
    /// block or list of content blocks (one user turn), or a list of steps
    /// (`user_input`, `model_output`, `function_call`, `function_result`; thoughts and
    /// built-in tool steps are skipped). `model` is absent for agent interactions.
    pub fn from_gemini_interactions(body: &Value) -> Result<Self, RequestError> {
        let obj = body.as_object().ok_or(RequestError::NotAnObject)?;
        let continuation_id = str_field(body, "previous_interaction_id");
        let mut messages = Vec::new();
        let user = |text: String| Message {
            role: "user".into(),
            text,
        };
        match obj.get("input") {
            Some(Value::String(text)) => messages.push(user(text.clone())),
            // A single content block.
            Some(block @ Value::Object(_)) => {
                messages.push(user(content_text(Some(&Value::Array(vec![block.clone()])))))
            }
            Some(Value::Array(items)) if items.iter().any(is_interaction_step) => {
                for step in items {
                    match step.get("type").and_then(Value::as_str) {
                        Some("user_input") => {
                            messages.push(user(content_text(step.get("content"))))
                        }
                        Some("model_output") => messages.push(Message {
                            role: "assistant".into(),
                            text: content_text(step.get("content")),
                        }),
                        Some("function_call") => messages.push(Message {
                            role: "assistant".into(),
                            text: tool_call_text(step.get("name"), step.get("arguments")),
                        }),
                        Some("function_result") => messages.push(user(match step.get("result") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => content_text(step.get("content")),
                        })),
                        _ => {}
                    }
                }
            }
            Some(Value::Array(_)) => messages.push(user(content_text(obj.get("input")))),
            None | Some(Value::Null) if continuation_id.is_some() => {}
            _ => return Err(RequestError::MissingInput("previous_interaction_id")),
        }
        Ok(Self {
            format: ApiFormat::GeminiInteractions,
            model: str_field(body, "model"),
            system: str_field(body, "system_instruction").filter(|s| !s.is_empty()),
            messages,
            has_tools: nonempty_array(body, "tools"),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            continuation_id,
            conversation: None,
        })
    }

    /// Rough prompt size in tokens (about 4 characters per token) for cost estimates.
    pub fn estimated_tokens(&self) -> u64 {
        let chars = self.system.as_ref().map_or(0, String::len)
            + self.messages.iter().map(|m| m.text.len()).sum::<usize>();
        (chars as u64).div_ceil(4)
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
                // Chat Completions / Anthropic / Gemini text, Open Responses text parts.
                Some("text" | "input_text" | "output_text") => {
                    p.get("text").and_then(Value::as_str).map(str::to_owned)
                }
                Some("refusal") => p.get("refusal").and_then(Value::as_str).map(str::to_owned),
                Some("tool_result") => Some(content_text(p.get("content"))),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Render a tool call as text so routers see what the agent did.
fn tool_call_text(name: Option<&Value>, arguments: Option<&Value>) -> String {
    let name = name.and_then(Value::as_str).unwrap_or("?");
    let args = match arguments {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    };
    format!("[tool call {name}: {args}]")
}

/// Whether a Gemini `input` list element is a step (rather than a content block).
fn is_interaction_step(v: &Value) -> bool {
    matches!(
        v.get("type").and_then(Value::as_str),
        Some("user_input" | "model_output" | "function_call" | "function_result" | "thought")
    ) || v
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| t.ends_with("_call") || t.ends_with("_result"))
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
    fn parses_open_responses_request() {
        let body = json!({
            "model": "auto",
            "instructions": "You are a coding agent.",
            "stream": true,
            "previous_response_id": "resp_123",
            "tools": [{"type": "function", "name": "bash"}],
            "input": [
                {"role": "developer", "content": "Be terse."},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run the tests"}]},
                {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{\"cmd\":\"cargo test\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "2 failed"},
                {"type": "reasoning", "summary": []},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Two tests fail."}]}
            ]
        });
        let r = RouteRequest::from_json(ApiFormat::OpenResponses, &body).unwrap();
        assert_eq!(
            r.system.as_deref(),
            Some("You are a coding agent.\n\nBe terse.")
        );
        let texts: Vec<_> = r
            .messages
            .iter()
            .map(|m| (m.role.as_str(), m.text.as_str()))
            .collect();
        assert_eq!(
            texts,
            [
                ("user", "run the tests"),
                ("assistant", "[tool call bash: {\"cmd\":\"cargo test\"}]"),
                ("user", "2 failed"),
                ("assistant", "Two tests fail."),
            ]
        );
        assert!(r.has_tools && r.stream);
        assert_eq!(r.continuation_id.as_deref(), Some("resp_123"));

        // String input, and continuation-only requests.
        let r = RouteRequest::from_open_responses(&json!({"model": "m", "input": "hi"})).unwrap();
        assert_eq!(r.last_user_text(), Some("hi"));
        assert!(
            RouteRequest::from_open_responses(&json!({"previous_response_id": "resp_1"})).is_ok()
        );
        assert_eq!(
            RouteRequest::from_open_responses(&json!({"model": "m"})),
            Err(RequestError::MissingInput("previous_response_id"))
        );
    }

    #[test]
    fn parses_gemini_interactions_request() {
        let body = json!({
            "model": "auto",
            "system_instruction": "You are helpful.",
            "previous_interaction_id": "int_9",
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "weather in Paris?"}]},
                {"type": "function_call", "id": "f1", "name": "get_weather", "arguments": {"city": "Paris"}},
                {"type": "function_result", "call_id": "f1", "result": {"temp": 18}},
                {"type": "thought", "content": []},
                {"type": "model_output", "content": [{"type": "text", "text": "18°C."}]},
                {"type": "user_input", "content": [{"type": "text", "text": "and Rome?"}]}
            ]
        });
        let r = RouteRequest::from_json(ApiFormat::GeminiInteractions, &body).unwrap();
        assert_eq!(r.system.as_deref(), Some("You are helpful."));
        let texts: Vec<_> = r
            .messages
            .iter()
            .map(|m| (m.role.as_str(), m.text.as_str()))
            .collect();
        assert_eq!(
            texts,
            [
                ("user", "weather in Paris?"),
                ("assistant", "[tool call get_weather: {\"city\":\"Paris\"}]"),
                ("user", "{\"temp\":18}"),
                ("assistant", "18°C."),
                ("user", "and Rome?"),
            ]
        );
        assert_eq!(r.continuation_id.as_deref(), Some("int_9"));

        // String input, a list of content blocks, and a single content block.
        let s = RouteRequest::from_gemini_interactions(&json!({"model": "m", "input": "Hello"}))
            .unwrap();
        assert_eq!(s.last_user_text(), Some("Hello"));
        let blocks = json!({"model": "m", "input": [{"type": "text", "text": "Describe"}, {"type": "image", "data": "..."}]});
        assert_eq!(
            RouteRequest::from_gemini_interactions(&blocks)
                .unwrap()
                .last_user_text(),
            Some("Describe")
        );
        let one = json!({"model": "m", "input": {"type": "text", "text": "One"}});
        assert_eq!(
            RouteRequest::from_gemini_interactions(&one)
                .unwrap()
                .last_user_text(),
            Some("One")
        );
        // Agent interactions have no model.
        let agent = RouteRequest::from_gemini_interactions(
            &json!({"agent": "deep-research", "input": "x"}),
        )
        .unwrap();
        assert_eq!(agent.model, None);
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
