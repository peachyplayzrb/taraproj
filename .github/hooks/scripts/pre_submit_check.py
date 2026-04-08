import json
import sys


def read_input():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def main():
    _event = read_input()
    print(
        json.dumps(
            {
                "continue": True,
                "systemMessage": (
                    "Before final response, include: files changed, what was verified, "
                    "and any unresolved verification gap."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
