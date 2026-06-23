"""Tests for the FastAPI query + live-stream server."""

from fastapi.testclient import TestClient

from xai.core.model import EngagementStatus, NodeDef, Topology
from xai.server.app import create_app
from xai.store.records import EngagementEnded, EngagementStarted, StepStarted
from xai.store.sqlite import SQLiteStore


def _seed(store, engagement_id="e1"):
    store.append(
        EngagementStarted(
            engagement_id=engagement_id,
            name="house_search",
            topology=Topology(nodes=[NodeDef(name="greet")]),
        )
    )
    store.append(StepStarted(engagement_id=engagement_id, step_id="s1", name="greet"))
    store.append(EngagementEnded(engagement_id=engagement_id, status=EngagementStatus.COMPLETED))


def test_list_engagements_endpoint():
    store = SQLiteStore()
    _seed(store)
    client = TestClient(create_app(store))

    resp = client.get("/api/engagements")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == "e1"
    assert body[0]["name"] == "house_search"
    assert body[0]["step_count"] == 1


def test_get_engagement_endpoint_returns_full_tree():
    store = SQLiteStore()
    _seed(store)
    client = TestClient(create_app(store))

    resp = client.get("/api/engagements/e1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "house_search"
    assert body["steps"][0]["name"] == "greet"
    assert body["topology"]["nodes"][0]["name"] == "greet"


def test_get_unknown_engagement_returns_404():
    store = SQLiteStore()
    client = TestClient(create_app(store))
    assert client.get("/api/engagements/nope").status_code == 404


def test_stream_endpoint_pushes_appended_records():
    store = SQLiteStore()
    client = TestClient(create_app(store))

    with client.websocket_connect("/api/stream") as ws:
        store.append(EngagementStarted(engagement_id="live1", name="streamed"))
        msg = ws.receive_json()
        assert msg["type"] == "engagement_started"
        assert msg["engagement_id"] == "live1"


def test_timeline_endpoint_returns_lanes():
    store = SQLiteStore()
    _seed(store)
    client = TestClient(create_app(store))

    resp = client.get("/api/engagements/e1/timeline")
    assert resp.status_code == 200
    body = resp.json()
    assert body["engagement_id"] == "e1"
    assert "total_ms" in body
    assert body["lanes"][0]["name"] == "greet"


def test_timeline_unknown_engagement_returns_404():
    store = SQLiteStore()
    client = TestClient(create_app(store))
    assert client.get("/api/engagements/nope/timeline").status_code == 404


def test_root_serves_viewer_html():
    store = SQLiteStore()
    client = TestClient(create_app(store))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
