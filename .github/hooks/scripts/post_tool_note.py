import json
import sys


def read_input():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def main():
    event = read_input()
    text = json.dumps(event).lower()

    behavior_markers = ["app.py", "templates/", "ipynb", "search", "ranking"]
    touched = any(marker in text for marker in behavior_markers)

    if touched:
        message = (
            "Behavior-impacting area detected. Remember to update "
            "00_admin/decision_log.md, 00_admin/change_log.md, and "
            "00_admin/backlog.md."
        )
    else:
        message = "Post-tool check complete."

    print(json.dumps({"systemMessage": message, "continue": True}))


if __name__ == "__main__":
    main()
