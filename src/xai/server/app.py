"""FastAPI app exposing the trace store to the viewer.

Endpoints
---------
* ``GET  /api/engagements``        -> list of engagement summaries
* ``GET  /api/engagements/{id}``   -> one full engagement tree (+ topology)
* ``WS   /api/stream``             -> records pushed as they are appended (live)
* ``GET  /``                       -> the static Cytoscape viewer
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..store.sqlite import SQLiteStore
from ..timeline import build_timeline

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(store) -> FastAPI:
    """Build a viewer app backed by ``store``."""
    app = FastAPI(title="xai viewer", version="0.0.1")

    @app.get("/api/engagements")
    def list_engagements():
        return [s.model_dump(mode="json") for s in store.list_engagements()]

    @app.get("/api/engagements/{engagement_id}")
    def get_engagement(engagement_id: str):
        try:
            return store.get_engagement(engagement_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown engagement") from exc

    @app.get("/api/engagements/{engagement_id}/timeline")
    def get_timeline(engagement_id: str):
        try:
            engagement = store.get_engagement(engagement_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown engagement") from exc
        return build_timeline(engagement).model_dump(mode="json")

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            async for record in store.subscribe():
                await ws.send_text(record.model_dump_json())
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="viewer")
    return app


def build_default_app() -> tuple[FastAPI, SQLiteStore]:
    """An in-memory store seeded with both demo runs, plus its app (for ``xai-server``)."""
    import asyncio

    from ..examples.demo_agent import run_demo
    from ..recorder import Recorder

    store = SQLiteStore()
    recorder = Recorder(store)

    async def _seed():
        await run_demo(recorder, route="search")
        await run_demo(recorder, route="escalate")

    asyncio.run(_seed())
    return create_app(store), store


def main() -> None:  # pragma: no cover - thin CLI entry point
    import os

    import uvicorn

    host = os.environ.get("XAI_HOST", "127.0.0.1")
    port = int(os.environ.get("XAI_PORT", "8400"))  # 8000 is commonly taken (Docker/backend)
    app, _ = build_default_app()
    print(f"xai viewer -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
