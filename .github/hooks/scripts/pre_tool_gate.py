import json
import re
import sys


def read_input():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def deny(reason: str):
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        "stopReason": reason,
    }
    print(json.dumps(payload))


def allow(message: str = ""):
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": message or "allowed",
        }
    }
    print(json.dumps(payload))


def main():
    event = read_input()
    text = json.dumps(event).lower()

    blocked_patterns = [
        r"git\s+reset\s+--hard",
        r"git\s+checkout\s+--",
        r"rm\s+-rf\s+",
        r"del\s+/s\s+/q\s+",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, text):
            deny("Blocked potentially destructive command by workspace policy")
            return

    allow("No blocked command patterns detected")


if __name__ == "__main__":
    main()
