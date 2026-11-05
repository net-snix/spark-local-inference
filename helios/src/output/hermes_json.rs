/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hermes-style calls: one `{"name": ..., "arguments": ...}` JSON object per tag pair.

use crate::Result;
use serde_json::Value;

pub struct HermesJson;
impl super::tagged::ToolDialect for HermesJson {
    const NAME: &'static str = "Hermes JSON";
    const TRIM_NEWLINES: bool = true;
    // ChatML templates join content and each call with one newline.
    const FIRST_CALL_SEPARATOR: &'static str = "\n";
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, _tools: &[Value]) -> Result<Vec<(String, String)>> {
        let call: Value = serde_json::from_str(body.trim())
            .map_err(|error| format!("malformed Hermes JSON tool call: {error}"))?;
        let Value::Object(mut call) = call else {
            return Err("Hermes tool call must be a JSON object".into());
        };
        let name = match call.remove("name") {
            Some(Value::String(name)) if !name.is_empty() => name,
            _ => return Err("tool function name is empty".into()),
        };
        // Templates render `arguments | tojson`, so an OpenAI-style string
        // argument reaches the model as a JSON string holding the object.
        let arguments = match call.remove("arguments") {
            Some(Value::Object(arguments)) => arguments,
            Some(Value::String(text)) => match serde_json::from_str(&text) {
                Ok(Value::Object(arguments)) => arguments,
                _ => return Err("Hermes tool call arguments string must encode a JSON object".into()),
            },
            _ => return Err("Hermes tool call requires object arguments".into()),
        };
        if !call.is_empty() {
            return Err("unexpected fields in Hermes tool call".into());
        }
        Ok(vec![(name, Value::Object(arguments).to_string())])
    }

    fn starts_call(text: &str) -> Option<bool> {
        // A call continues with its JSON object; a tag followed by prose remains text.
        let text = text.strip_prefix(Self::OPEN).unwrap_or(text).trim_start();
        text.chars().next().map(|first| first == '{')
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::contract::{ReasoningParser, ReasoningParserKind, ToolParser, ToolParserKind};
    use crate::output::tagged::{OutputConfig as TaggedConfig, OutputStart, ToolDialect};
    use crate::output::{Input, OutputConfig, OutputPipeline, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;

    fn syntax() -> crate::model::contract::FrontendContract {
        let mut syntax = crate::model::testing::metadata().frontend;
        syntax.tools = Some(ToolParser {
            kind: ToolParserKind::HermesJson,
            wrapping_newlines: 0,
            tag_separator: String::new(),
        });
        syntax.reasoning = Some(ReasoningParser {
            kind: ReasoningParserKind::ThinkTags,
            open: "<think>".into(),
            close: "</think>".into(),
            wrapping_newlines: 1,
            terminal_markers: vec![],
        });
        syntax
    }

    fn tools() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"get_weather"}}),
            json!({"type":"function","function":{"name":"get_time"}}),
        ]
    }

    /// Feeds `pieces` as consecutive decoded deltas and returns the message.
    fn stream(pieces: &[&str]) -> Result<Value> {
        let mut parser = OutputPipeline::new(OutputConfig {
            syntax: syntax(),
            control_tokens: vec![],
            content: TaggedConfig {
                structured: false,
                start: OutputStart::Content,
                tools: tools(),
            },
        });
        let mut output = Accumulator::new("hermes".into());
        for piece in pieces {
            output.apply(parser.push(Input::Text(piece))?)?;
        }
        output.apply(parser.finish(Termination::Eos)?)?;
        output.finish()?;
        Ok(output.message())
    }

    /// Every split of `text` into one, two, and three deltas at character
    /// boundaries, plus one character per delta.
    fn every_split(text: &str, check: impl Fn(&Value, &str)) {
        let bounds: Vec<usize> = text.char_indices().map(|(at, _)| at).skip(1).collect();
        check(&stream(&[text]).unwrap(), "whole");
        let characters: Vec<String> = text.chars().map(String::from).collect();
        let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
        check(&stream(&characters).unwrap(), "characters");
        for (index, &first) in bounds.iter().enumerate() {
            check(&stream(&[&text[..first], &text[first..]]).unwrap(), &format!("{first}"));
            for &second in &bounds[index + 1..] {
                let pieces = [&text[..first], &text[first..second], &text[second..]];
                check(&stream(&pieces).unwrap(), &format!("{first},{second}"));
            }
        }
    }

    fn content(message: &Value) -> &str {
        message["content"].as_str().unwrap_or_default()
    }

    #[test]
    fn single_call_streams_at_every_boundary() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>";
        every_split(text, |message, split| {
            assert_eq!(content(message), "", "split {split}");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 1, "split {split}");
            assert_eq!(calls[0]["function"]["name"], "get_weather", "split {split}");
            assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Oslo\"}", "split {split}");
        });
    }

    #[test]
    fn consecutive_calls_after_reasoning_and_text_stream_at_every_boundary() {
        let text = "<think>\nplan\n</think>\n\nChecking café.\n<tool_call>\n{\"name\": \"get_weather\", \
                    \"arguments\": {\"city\": \"Tromsø\", \"days\": [1, 2]}}\n</tool_call>\n<tool_call>\n\
                    {\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>";
        every_split(text, |message, split| {
            assert_eq!(message["reasoning_content"], "plan", "split {split}");
            assert_eq!(content(message), "Checking café.", "split {split}");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2, "split {split}");
            assert_eq!(calls[0]["function"]["name"], "get_weather", "split {split}");
            assert_eq!(
                calls[0]["function"]["arguments"],
                "{\"city\":\"Tromsø\",\"days\":[1,2]}",
                "split {split}"
            );
            assert_eq!(calls[1]["function"]["name"], "get_time", "split {split}");
            assert_eq!(calls[1]["function"]["arguments"], "{}", "split {split}");
            assert_ne!(calls[0]["id"], calls[1]["id"], "split {split}");
        });
    }

    #[test]
    fn string_arguments_decode_to_their_object() {
        let text = "Sure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                    \"{\\\"city\\\": \\\"Oslo\\\", \\\"units\\\": {\\\"temp\\\": \\\"C\\\"}}\"}\n</tool_call>";
        every_split(text, |message, split| {
            assert_eq!(content(message), "Sure.", "split {split}");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\":\"Oslo\",\"units\":{\"temp\":\"C\"}}",
                "split {split}"
            );
        });
    }

    #[test]
    fn tag_mentions_in_prose_remain_content() {
        let text = "Wrap calls in <tool_call> tags.";
        every_split(text, |message, split| {
            assert_eq!(content(message), text, "split {split}");
            assert!(message.get("tool_calls").is_none_or(Value::is_null), "split {split}");
        });
    }

    #[test]
    fn malformed_json_fails_the_output() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": }}\n</tool_call>";
        assert!(stream(&[text]).is_err());
        let characters: Vec<String> = text.chars().map(String::from).collect();
        let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
        assert!(stream(&characters).is_err());
        assert!(stream(&["<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}"]).is_err());
    }

    #[test]
    fn calls_require_one_named_object_with_object_arguments() {
        let parse = |body: &str| HermesJson::parse(body, &[]);
        assert_eq!(
            parse("\n{\"arguments\": {\"b\": 1, \"a\": [true, null]}, \"name\": \"f\"}\n").unwrap(),
            vec![("f".to_owned(), "{\"b\":1,\"a\":[true,null]}".to_owned())]
        );
        for body in [
            "",
            "[]",
            "{\"name\": \"f\", \"arguments\": {}",
            "{\"name\": \"f\", \"arguments\": {}} {\"name\": \"g\", \"arguments\": {}}",
            "{\"arguments\": {}}",
            "{\"name\": \"\", \"arguments\": {}}",
            "{\"name\": 3, \"arguments\": {}}",
            "{\"name\": \"f\"}",
            "{\"name\": \"f\", \"arguments\": [1]}",
            "{\"name\": \"f\", \"arguments\": \"Oslo\"}",
            "{\"name\": \"f\", \"arguments\": \"[1]\"}",
            "{\"name\": \"f\", \"arguments\": \"{\\\"a\\\":\"}",
            "{\"name\": \"f\", \"arguments\": {}, \"id\": \"x\"}",
        ] {
            assert!(parse(body).is_err(), "{body}");
        }
    }
}
