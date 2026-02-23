from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import SignalGate


Factory = Callable[[dict[str, Any]], SignalGate]


_GATE_REGISTRY: dict[str, Factory] = {}


def register_gate(name: str) -> Callable[[type], type]:
    """Decorator to register a gate class by name.

    The decorated class must be instantiable via **params.
    """

    def wrap(cls: type) -> type:
        if name in _GATE_REGISTRY:
            raise KeyError(f"Gate already registered: {name!r}")

        def _factory(params: dict[str, Any]) -> SignalGate:
            return cls(**(params or {}))  # type: ignore[arg-type]

        _GATE_REGISTRY[name] = _factory
        return cls

    return wrap


def register_factory(name: str, factory: Factory) -> None:
    if name in _GATE_REGISTRY:
        raise KeyError(f"Gate already registered: {name!r}")
    _GATE_REGISTRY[name] = factory


@dataclass(frozen=True)
class GateSpec:
    gate: str
    params: dict[str, Any] | None = None


def make_gate(spec: GateSpec | dict[str, Any]) -> SignalGate:
    if isinstance(spec, dict):
        gate = str(spec.get("gate") or spec.get("name") or "").strip()
        params = spec.get("params")
        if params is None:
            # allow flat style: {gate: "ema_sep", ema_fast: 40, ...}
            params = {k: v for k, v in spec.items() if k not in {"gate", "name", "params"}}
        if not gate:
            raise ValueError(f"Invalid gate spec: {spec!r}")
        spec = GateSpec(gate=gate, params=params)

    name = spec.gate
    if name not in _GATE_REGISTRY:
        raise KeyError(f"Unknown gate: {name!r}. Registered: {sorted(_GATE_REGISTRY)}")

    return _GATE_REGISTRY[name](dict(spec.params or {}))


def registered_gates() -> list[str]:
    return sorted(_GATE_REGISTRY.keys())
