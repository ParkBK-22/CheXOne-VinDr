import re


def normalize_answer(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s/_.-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_yes_no(text: str) -> str:
    ans = normalize_answer(text)

    if ans.startswith("yes") or ans in {"y", "true", "present"}:
        return "yes"

    if ans.startswith("no") or ans in {"n", "false", "absent"}:
        return "no"

    if "yes" in ans and "no" not in ans:
        return "yes"

    if "no" in ans and "yes" not in ans:
        return "no"

    return "unknown"
