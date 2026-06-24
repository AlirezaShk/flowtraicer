"""FastAPI query + live-stream API and the static viewer."""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
