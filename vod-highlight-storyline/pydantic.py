"""Minimal local fallback for offline test environments."""

from __future__ import annotations


def Field(default=None, default_factory=None):
    if default_factory is not None:
        return default_factory()
    return default


class BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if not hasattr(self, name):
                setattr(self, name, value)

    def model_dump(self):
        out = {}
        for name in dir(self):
            if name.startswith("_"):
                continue
            value = getattr(self, name)
            if callable(value):
                continue
            out[name] = value
        return out
