"""Pure text helpers."""


def slugify(text: str) -> str:
    return "-".join(text.lower().split())


def is_palindrome(text: str) -> bool:
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]


def word_count(text: str) -> int:
    return len(text.split())
