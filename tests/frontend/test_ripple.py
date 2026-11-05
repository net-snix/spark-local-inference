# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Ripple's annotated ChatML template against the checkpoint's Transformers rendering."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ignition.preparation.frontend import annotated_template_asset
from ignition.recipes.ripple.recipe import RECIPE

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "ignition/recipes/ripple/assets/chat_template.patch.json"
MARK = "{%- mark -%}"
SENTINEL = ""
END = "<|im_end|>\n"
USER = dict(role="user", content="What is the weather in Paris?")
TOOLS = [dict(type="function", function=dict(name="lookup", description="Search documents.",
    parameters=dict(type="object", properties=dict(query=dict(type="string"), count=dict(type="integer")),
                    required=["query"]))),
         dict(type="function", function=dict(name="execute_bash", description="Run a shell command.",
    parameters=dict(type="object", properties=dict(command=dict(type="string")), required=["command"])))]


def snapshot():
    checkpoint = RECIPE.checkpoints[0]
    candidates = [Path(os.environ["FLARE_RIPPLE_CHECKPOINT"])] if os.environ.get("FLARE_RIPPLE_CHECKPOINT") else []
    candidates.append(Path.home() / ".cache/huggingface/hub" / f"models--{checkpoint.repository.replace('/', '--')}"
                      / "snapshots" / checkpoint.revision)
    for path in candidates:
        if all((path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")):
            return path
    pytest.skip("set FLARE_RIPPLE_CHECKPOINT or cache the pinned Ripple tokenizer and chat template")


def tool_calls(arguments):
    return [dict(id="call_1", type="function", function=dict(name="lookup", arguments=arguments[0])),
            dict(id="call_2", type="function", function=dict(name="execute_bash", arguments=arguments[1]))]


def cases():
    system = dict(role="system", content="You are a careful assistant.")
    reasoning = dict(role="assistant", content="Paris is sunny.", reasoning_content="Recall the forecast.")
    objects = ({"query": "Paris\nweather", "count": 2}, {"command": "echo 'héllo' && pwd"})
    strings = tuple(json.dumps(value) for value in objects)
    rounds = [system, USER, dict(role="assistant", content="", reasoning_content="Check both.",
                                 tool_calls=tool_calls(objects)),
              dict(role="tool", tool_call_id="call_1", content='{"temperature_c": 21}'),
              dict(role="tool", tool_call_id="call_2", content="/home/user"),
              dict(role="assistant", content="It is 21 °C."), dict(role="user", content="Thanks!")]
    yield "plain", [USER], {}
    yield "plain_no_generation", [USER], dict(add_generation_prompt=False)
    yield "system_user", [system, USER], {}
    yield "reasoning_history", [system, USER, reasoning, dict(role="user", content="And tomorrow?")], {}
    yield "reasoning_cleared", [system, USER, reasoning, dict(role="user", content="And tomorrow?")], dict(
        clear_thinking=True)
    yield "reasoning_inline", [USER, dict(role="assistant", content="<think>\nInline.\n</think>\nDone.",
                                          reasoning_content="Ignored."), USER], {}
    yield "empty_messages", [dict(role="user", content=""), dict(role="assistant", content=None),
                             dict(role="user", content="Again")], {}
    yield "tools_without_system", [USER], dict(tools=TOOLS)
    yield "tools_with_system", [system, USER], dict(tools=TOOLS)
    yield "tool_calls_object", rounds, dict(tools=TOOLS)
    yield "tool_calls_string", [*rounds[:2], dict(rounds[2], tool_calls=tool_calls(strings)), *rounds[3:]], dict(
        tools=TOOLS)
    yield "tool_calls_content", [USER, dict(role="assistant", content="Let me check.",
                                            tool_calls=tool_calls(objects)[:1]),
                                 dict(role="tool", tool_call_id="call_1", content="sunny")], dict(tools=TOOLS)
    yield "pending_tool_results", rounds[:5], dict(tools=TOOLS)
    yield "later_system", [USER, dict(role="assistant", content="Hi."), system, USER], {}
    yield "utf8", [dict(role="user", content="中文 é á 👨‍👩‍👧‍👦\r\n  next line\t"),
                   dict(role="assistant", content="Ça va 😀", reasoning_content="Ünïcödé 思考"),
                   dict(role="user", content="  leading and trailing  \n\n")], {}
    yield "continuation", [USER, dict(role="assistant", content="Partial answer")], dict(
        add_generation_prompt=False, continue_final_message=True)


def expected_marks(tokenizer, messages, options, rendered):
    """Byte endpoints of the preamble and of every normalized message, from prefix renderings."""
    data = rendered.encode()
    if options.get("tools"):
        preamble = data.index(END.encode(), data.index(b"\n</tools>")) + len(END)
    elif messages[0]["role"] == "system":
        preamble = data.index(END.encode()) + len(END)
    else:
        preamble = 0
    marks = {preamble}
    for count in range(1, len(messages) + 1):
        prefix = tokenizer.apply_chat_template(messages[:count], tokenize=False,
            **{**options, "add_generation_prompt": False}).encode()
        if not data.startswith(prefix):
            # A tool result followed by another stays inside the merged user turn without its closing marker.
            assert messages[count - 1]["role"] == "tool" and prefix.endswith(END.encode())
            prefix = prefix[:-len(END)]
            assert data.startswith(prefix)
        marks.add(len(prefix))
    return sorted(marks)


def render_marked(tokenizer, template, messages, options):
    marked = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, **options)
    text, marks = "", []
    for part in marked.split(SENTINEL)[:-1]:
        text += part
        marks.append(len(text.encode()))
    return text + marked.split(SENTINEL)[-1], marks


def token_endpoints(tokenizer, text, marks):
    """Exact token cuts at byte marks: every earlier token ends at or before it and every later one starts after."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(len(text[:start].encode()), len(text[:end].encode())) for start, end in encoding["offset_mapping"]]
    cuts = []
    for mark in marks:
        count = sum(end <= mark for _, end in offsets)
        if all(start >= mark for start, _ in offsets[count:]) and 0 < count < len(offsets):
            cuts.append(count)
    return encoding["input_ids"], sorted(set(cuts))


def load():
    from transformers import AutoTokenizer

    source = snapshot()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    template = (source / "chat_template.jinja").read_bytes()
    return source, tokenizer, template, annotated_template_asset(template, PATCH)


def test_reference_tokenizer_uses_the_patched_template_source():
    _, tokenizer, template, asset = load()
    assert asset["text"].encode().replace(MARK.encode(), b"") == template
    assert asset["text"].count(MARK) == 2
    # tokenizer_config.json carries an older tool-less ChatML template; chat_template.jinja takes precedence.
    assert tokenizer.chat_template == template.decode()
    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"]) == [100264, 100265]


@pytest.mark.parametrize("name,messages,options", list(cases()), ids=[case[0] for case in cases()])
def test_annotated_template_matches_reference_rendering(name, messages, options):
    _, tokenizer, _, asset = load()
    options = {"add_generation_prompt": True, **options}
    reference = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    text, marks = render_marked(tokenizer, asset["text"].replace(MARK, '{{- "%s" -}}' % SENTINEL), messages,
                                options)
    assert text == reference
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    assert ids == tokenizer.encode(text, add_special_tokens=False)
    if name == "continuation":
        assert text.endswith("<|im_start|>assistant\nPartial answer")
        return
    # Marks follow the preamble and each loop iteration; a leading system message adds a duplicate.
    assert sorted(set(marks)) == expected_marks(tokenizer, messages, options, reference)
    data = reference.encode()
    for mark in marks:
        assert mark == 0 or data[:mark].endswith((END.encode(), b"</tool_response>"))
    # Every completed turn ends at an exact token cut.
    _, cuts = token_endpoints(tokenizer, reference, [m for m in marks if data[:m].endswith(END.encode())])
    closing = {m for m in marks if 0 < m < len(data) and data[:m].endswith(END.encode())}
    assert len(cuts) == len(closing)


def native_fixture(source, tokenizer, asset, directory):
    """Write the native frontend's assets and reference cases; return the fixture path."""
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(source / name, directory / name)
    (directory / "chat_template.jinja").write_text(asset["text"])
    fixtures = dict(cases=[], encode_cases=[], decode_cases=[], assets={},
                    template_defaults={"add_generation_prompt": True}, default_overrides=[["plain", {}]])
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        fixtures["assets"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    template = asset["text"].replace(MARK, '{{- "%s" -}}' % SENTINEL)
    for name, messages, options in cases():
        options = {"add_generation_prompt": True, **options}
        rendered, marks = render_marked(tokenizer, template, messages, options)
        ids, cuts = token_endpoints(tokenizer, rendered, marks)
        case = dict(name=name, messages=messages, options=options, rendered=rendered, token_ids=ids)
        if name != "continuation":
            case["message_boundaries"] = cuts
        fixtures["cases"].append(case)
    for index, text in enumerate(["Hello, world!", "中文 é á 👨‍👩‍👧‍👦\r\n",
                                  "<think>\nwork\n</think>\nanswer<|im_end|>",
                                  '<tool_call>\n{"name": "lookup", "arguments": {"query": "Paris"}}\n</tool_call>']):
        ids = tokenizer.encode(text, add_special_tokens=False)
        fixtures["encode_cases"].append(dict(name=f"encode_{index}", text=text, add_special_tokens=False,
                                             token_ids=ids))
        for skip in (False, True):
            fixtures["decode_cases"].append(dict(name=f"decode_{index}_{skip}", token_ids=ids,
                skip_special_tokens=skip, text=tokenizer.decode(ids, skip_special_tokens=skip)))
    path = directory / "cases.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False))
    return path, fixtures


def test_native_ripple_prompts_match_transformers(tmp_path):
    binary = ROOT / "helios/target/debug/test_frontend"
    if not binary.is_file():
        pytest.skip("build test_frontend with frontend,test-tools features")
    source, tokenizer, _, asset = load()
    path, fixtures = native_fixture(source, tokenizer, asset, tmp_path)
    result = subprocess.run([str(binary), str(tmp_path), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(cases=len(fixtures["cases"]), encode_cases=len(fixtures["encode_cases"]),
                                             decode_cases=len(fixtures["decode_cases"]), default_override_cases=1)
