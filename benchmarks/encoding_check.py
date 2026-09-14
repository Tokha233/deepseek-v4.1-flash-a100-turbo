"""Compare a serving encoder with the checkpoint's independent reference."""

import importlib.util
import json
from pathlib import Path

import typer


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(reference: Path, candidate: Path, output: Path) -> None:
    ref, impl = load(reference, "reference_encoding"), load(candidate, "candidate_encoding")
    tool = {"type": "function", "function": {
        "name": "lookup", "description": "查找值", "parameters": {
            "type": "object", "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }}
    user = {"role": "user", "content": "查找 alpha 和 beta。"}
    history = [
        {"role": "system", "content": "Use tools to verify facts.", "tools": [tool]},
        user,
        {"role": "assistant", "content": "", "reasoning_content": "I should verify both values.",
         "tool_calls": [{"id": name, "type": "function", "function": {
             "name": "lookup", "arguments": json.dumps({"key": name}),
         }} for name in ("alpha", "beta")]},
        {"role": "tool", "tool_call_id": "beta", "content": "29"},
        {"role": "tool", "tool_call_id": "alpha", "content": "17"},
    ]
    cases = {
        "user": [user],
        "system": [{"role": "system", "content": "Be concise."}, user],
        "multiturn": [user, {"role": "assistant", "content": "17 and 29.",
                                "reasoning_content": "Verified values."},
                      {"role": "user", "content": "Compute their sum."}],
        "mid_system": [user, {"role": "assistant", "content": "Done."},
                       {"role": "system", "content": "Answer in Chinese."}],
        "tools_first": history[:2],
        "tools_reordered_results": history,
        "tools_followup": history + [{"role": "assistant", "content": "46",
                                      "reasoning_content": "17 + 29 = 46."},
                                     {"role": "user", "content": "Explain how."}],
    }
    results = []
    for name, messages in cases.items():
        for mode in ("chat", "thinking"):
            for effort in (None, 1, 50, 75, 100, "low", "high", "max"):
                kwargs = {"thinking_mode": mode, "reasoning_effort": effort}
                expected, actual = ref.encode_messages(messages, **kwargs), impl.encode_messages(messages, **kwargs)
                row = {"case": name, **kwargs, "match": expected == actual}
                if expected != actual:
                    row.update(expected=expected, actual=actual)
                results.append(row)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(results), "matched": sum(r["match"] for r in results),
                      "mismatches": [{k: r[k] for k in ("case", "thinking_mode", "reasoning_effort")}
                                     for r in results if not r["match"]]}, ensure_ascii=False))


if __name__ == "__main__":
    typer.run(main)
