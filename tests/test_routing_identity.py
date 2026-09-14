from routing_identity import conversation_key


def test_tool_rounds_keep_identity_and_distinct_tasks_remain_separate() -> None:
    initial = {"model": "deepseek-v41", "messages": [
        {"role": "system", "content": "Fix the repository."},
        {"role": "user", "content": "Fix issue 18."},
    ]}
    continued = initial | {"messages": initial["messages"] + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "The test failed."},
    ], "stream": True, "temperature": 1}
    assert conversation_key(initial) == conversation_key(continued)
    assert conversation_key(initial) != conversation_key(initial | {
        "messages": initial["messages"][:1] + [{"role": "user", "content": "Fix issue 19."}]})
    assert conversation_key(initial, "token-a") != conversation_key(initial, "token-b")
    assert conversation_key(initial) != conversation_key(initial | {"model": "another-model"})


def test_structured_content_and_missing_conversation_identity() -> None:
    initial = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Read this file."}]}]}
    reordered = {"messages": [{"content": [{"text": "Read this file.", "type": "text"}], "role": "user"}]}
    assert conversation_key(initial) == conversation_key(reordered)
    assert not conversation_key({"prompt": [1, 2, 3]})
    assert not conversation_key({"messages": [{"role": "system", "content": "Instructions only."}]})
    assert not conversation_key({"messages": [None]})
