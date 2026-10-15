"""A lazily created singleton."""


class Registry:
    _instance = None

    def __init__(self) -> None:
        self.mode = "normal"

    @classmethod
    def get(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = Registry()
        return cls._instance
