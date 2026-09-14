"""Exercise chunk boundaries, sparse retrieval, and repeat-prefix requests."""

import hashlib
import json
import time
from pathlib import Path

import httpx
import typer
from tokenizers import Tokenizer

from encoding_check import load


def main(model: Path, output: Path, endpoint: str = "http://127.0.0.1:8083",
         lengths: list[int] = [2048, 8192, 24576], reasoning_effort: int = 75,
         max_tokens: int = 1024) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    encoder = load(model / "encoding/encoding.py", "reference_encoding")
    noise = "\n".join(f"Record {i:06d}: ordinary archive entry, status unchanged, no access code here."
                      for i in range(max(lengths)))
    noise_ids = tokenizer.encode(noise, add_special_tokens=False).ids
    results = []
    with httpx.Client(base_url=endpoint, timeout=1800, trust_env=False) as client:
        for length in lengths:
            code = f"ORCHID-{length}-7391"
            text = (tokenizer.decode(noise_ids[:length // 2]) + f"\nThe unique access code is {code}.\n"
                    + tokenizer.decode(noise_ids[length // 2:length])
                    + "\nReturn the unique access code exactly. Ignore the ordinary archive records.")
            messages = [{"role": "user", "content": text}]
            request = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "messages": messages,
                       "temperature": 0, "max_tokens": max_tokens,
                       "cache_salt": hashlib.sha256(f"{output.resolve()}/{length}".encode()).hexdigest(),
                       "chat_template_kwargs": {"thinking": True, "reasoning_effort": reasoning_effort}}
            (output / f"{length}-request.json").write_text(json.dumps(request))
            prompt = encoder.encode_messages(messages, thinking_mode="thinking", reasoning_effort=reasoning_effort)
            expected_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            tokenized = client.post("/tokenize", json={k: request[k] for k in ("model", "messages", "chat_template_kwargs")})
            (output / f"{length}-tokenized.json").write_text(tokenized.text)
            tokenized.raise_for_status()
            assert tokenized.json()["tokens"] == expected_ids
            for repeat in range(2):
                started = time.monotonic()
                response = client.post("/v1/chat/completions", json=request)
                elapsed = time.monotonic() - started
                (output / f"{length}-{repeat}-response.json").write_text(response.text)
                response.raise_for_status()
                data = response.json()
                assert code in (data["choices"][0]["message"].get("content") or ""), data
                assert data["choices"][0]["finish_reason"] != "length"
                results.append({"requested_noise_tokens": length, "encoded_prompt_tokens": len(expected_ids),
                                "repeat": repeat, "seconds": elapsed, "usage": data["usage"]})
                (output / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    typer.run(main)
