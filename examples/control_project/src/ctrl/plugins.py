"""A plugin list filled at runtime."""

PLUGINS: list[str] = []


def register(name: str) -> None:
    if name not in PLUGINS:
        PLUGINS.append(name)
