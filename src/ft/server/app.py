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
    app = FastAPI(title="FlowTraicer viewer", version="0.0.1")

    @app.get("/api/engagements")
    def list_engagements():
        """List one summary row per engagement."""
        return [s.model_dump(mode="json") for s in store.list_engagements()]

    @app.get("/api/engagements/{engagement_id}")
    def get_engagement(engagement_id: str):
        """Return one full engagement tree (steps, events, topology)."""
        try:
            return store.get_engagement(engagement_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown engagement") from exc

    @app.get("/api/engagements/{engagement_id}/timeline")
    def get_timeline(engagement_id: str):
        """Return the temporal viewmodel (lanes + marks) for the timeline view."""
        try:
            engagement = store.get_engagement(engagement_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown engagement") from exc
        return build_timeline(engagement).model_dump(mode="json")

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        """Push records to the client as they are appended (live monitoring)."""
        await ws.accept()
        try:
            async for record in store.subscribe():
                await ws.send_text(record.model_dump_json())
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="viewer")
    return app


def build_default_app() -> tuple[FastAPI, SQLiteStore]:
    """An in-memory store seeded with both demo runs, plus its app (for ``ft-server``)."""
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


def serve(store, *, host: str = "127.0.0.1", port: int = 8400) -> None:
    """Run the viewer for ``store`` — the one-liner way to look at your traces.

    FlowTraicer owns the uvicorn server, so you don't build an app or wire uvicorn yourself::

        from ft.server.app import serve          # (also: from ft.server import serve)
        from ft.store.postgres import PostgresStore

        serve(PostgresStore(DSN), host="0.0.0.0", port=8400)

    Blocks until the server stops. Bind to ``0.0.0.0`` only behind your own auth/network controls —
    traces can contain user data.
    """
    import uvicorn

    print(f"FlowTraicer viewer -> http://{host}:{port}")
    uvicorn.run(create_app(store), host=host, port=port)


def main() -> None:  # pragma: no cover - thin CLI entry point
    """Serve the demo-seeded viewer (the ``ft-server`` console script)."""
    import os

    host = os.environ.get("FT_HOST", os.environ.get("XAI_HOST", "127.0.0.1"))
    port = int(os.environ.get("FT_PORT", os.environ.get("XAI_PORT", "8400")))
    _, store = build_default_app()
    serve(store, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
