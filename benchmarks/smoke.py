"""Exercise a real V4.1 API, retaining responses and streamed wire data."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import typer

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def main(output: Path, endpoint: str = "http://127.0.0.1:8083", concurrency: int = 4,
         occupier: str = "unknown") -> None:
    output.mkdir(parents=True, exist_ok=False)
    with httpx.Client(base_url=endpoint, timeout=600, trust_env=False) as client:
        def chat(name: str, messages: list[dict], **kwargs) -> dict:
            request = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 1024,
                       "chat_template_kwargs": {"thinking": True, "reasoning_effort": 75}, **kwargs}
            (output / (name + "-request.json")).write_text(json.dumps(request, ensure_ascii=False, indent=2))
            started = time.monotonic()
            response = client.post("/v1/chat/completions", json=request)
            (output / (name + "-response.json")).write_text(response.text)
            response.raise_for_status()
            data = response.json()
            assert data["choices"][0]["finish_reason"] != "length", data
            (output / (name + "-timing.json")).write_text(json.dumps({"seconds": time.monotonic() - started,
                                                                       "usage": data.get("usage")}))
            return data

        response = client.get("/v1/models")
        response.raise_for_status()
        (output / "models.json").write_text(response.text)
        assert MODEL in [m["id"] for m in response.json()["data"]]
        messages = [{"role": "user", "content": "What is 17 * 19? Return only the integer in your final answer."}]
        for effort in (1, 50, 75, 100):
            result = chat(f"effort-{effort}", messages,
                          chat_template_kwargs={"thinking": True, "reasoning_effort": effort})
            assert re.search(r"\b323\b", result["choices"][0]["message"].get("content") or ""), result

        tools = [{"type": "function", "function": {
            "name": "lookup", "description": "Read a value from the local experiment table.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string", "enum": ["alpha", "beta"]}},
                           "required": ["key"], "additionalProperties": False},
        }}]
        history = [{"role": "system", "content": "Use lookup to obtain both values before computing the sum. Do not guess."},
                   {"role": "user", "content": "Get alpha and beta, then report their sum as an integer."}]
        seen = set()
        for turn in range(4):
            result = chat(f"tools-{turn}", history, tools=tools, parallel_tool_calls=True)
            message = result["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            history.append({k: v for k, v in message.items()
                            if k in ("role", "content", "reasoning_content", "reasoning", "tool_calls") and v is not None})
            if not calls:
                assert seen == {"alpha", "beta"} and re.search(r"\b46\b", message.get("content") or ""), result
                break
            for call in calls:
                assert call["function"]["name"] == "lookup"
                key = json.loads(call["function"]["arguments"])["key"]
                seen.add(key)
                history.append({"role": "tool", "tool_call_id": call["id"],
                                "content": str({"alpha": 17, "beta": 29}[key])})
        else:
            raise AssertionError("Tool conversation did not complete")
        (output / "tool-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))

        def arithmetic(index: int) -> dict:
            a, b = index + 13, index + 29
            result = chat(f"concurrent-{index}", [{"role": "user", "content": f"Compute {a}+{b}. Return only the integer."}])
            assert re.search(rf"\b{a+b}\b", result["choices"][0]["message"].get("content") or ""), result
            return result

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            concurrent = list(pool.map(arithmetic, range(concurrency * 2)))
        elapsed = time.monotonic() - started
        request = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 1024,
                   "chat_template_kwargs": {"thinking": True, "reasoning_effort": 75}, "stream": True,
                   "stream_options": {"include_usage": True}}
        chunks, content, usage = [], [], None
        with client.stream("POST", "/v1/chat/completions", json=request) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines():
                chunks.append(line)
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices", []):
                        content.append(choice.get("delta", {}).get("content") or "")
        (output / "stream.sse").write_text("\n".join(chunks))
        assert re.search(r"\b323\b", "".join(content)) and usage and "data: [DONE]" in chunks
        summary = {"efforts": [1, 50, 75, 100], "tools": "passed", "stream": "passed",
                   "concurrent_requests": len(concurrent), "concurrency": concurrency,
                   "seconds": elapsed, "output_tokens": sum(r["usage"]["completion_tokens"] for r in concurrent),
                   "occupier": occupier, "note": "Correctness smoke; use benchmark.py for throughput."}
        summary["output_tokens_per_second"] = summary["output_tokens"] / elapsed
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary))


if __name__ == "__main__":
    typer.run(main)
