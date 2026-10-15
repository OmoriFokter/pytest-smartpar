"""Files in <project>/output (shared by every process that runs the tests)."""

import time

from ctrl.paths import OUTPUT


def write_text(name: str, content: str, append: bool = False) -> None:
    OUTPUT.mkdir(exist_ok=True)
    with open(OUTPUT / name, "a" if append else "w") as fh:
        # Two chunks with a pause: a concurrent reader can see a half-written file.
        half = len(content) // 2
        fh.write(content[:half])
        fh.flush()
        time.sleep(0.02)
        fh.write(content[half:])


def read_text(name: str) -> str | None:
    try:
        with open(OUTPUT / name) as fh:
            return fh.read()
    except FileNotFoundError:
        return None


def read_text_if_exists(name: str) -> str | None:
    """Reads only when the file is already there (a conditional access)."""
    path = OUTPUT / name
    if not path.exists():
        return None
    return path.read_text()
