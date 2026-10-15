"""OD pair (singleton attribute) and a brittle test (plugin list, initialization flag)."""

from ctrl import config, plugins
from ctrl.registry import Registry


def test_registry_default_mode():
    assert Registry.get().mode == "normal"


def test_enable_debug_mode():
    Registry.get().mode = "debug"  # never restored


def test_registry_with_monkeypatch(monkeypatch):
    monkeypatch.setattr(Registry, "_instance", Registry())
    Registry.get().mode = "test"  # on the temporary instance only


def test_register_csv_plugin():
    plugins.register("csv")


def test_csv_plugin_available():
    assert "csv" in plugins.PLUGINS


def test_initialize():
    config.INITIALIZED = True


def test_requires_initialized():
    assert config.INITIALIZED
