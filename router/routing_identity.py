import hashlib
import json


def conversation_key(data: dict, authorization: str = "") -> str:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return ""
    prefix = []
    for message in messages:
        if not isinstance(message, dict):
            return ""
        prefix.append(message)
        if message.get("role") == "user":
            material = {"model": data.get("model"), "messages": prefix}
            encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return "chat-" + hashlib.sha256((authorization + "\0" + encoded).encode()).hexdigest()
    return ""
