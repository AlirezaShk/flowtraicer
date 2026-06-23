"""Read the static workflow topology from a compiled LangGraph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from xai.core.model import EdgeDef, NodeDef, Topology

# LangGraph's synthetic entry/exit nodes — not part of the authored workflow.
_SYNTHETIC = {"__start__", "__end__"}


def read_topology(
    compiled,
    *,
    global_nodes: Iterable[str] = (),
    node_tools: Mapping[str, list[str]] | None = None,
) -> Topology:
    """Build a :class:`Topology` from a compiled LangGraph's ``get_graph()``.

    ``global_nodes`` are flagged as re-routing nodes; ``node_tools`` supplies the
    per-step tool list the graph itself doesn't expose.
    """
    global_set = set(global_nodes)
    tools_map = dict(node_tools or {})
    graph = compiled.get_graph()

    nodes = [
        NodeDef(name=nid, is_global=nid in global_set, tools=tools_map.get(nid, []))
        for nid in graph.nodes
        if nid not in _SYNTHETIC
    ]
    edges = [
        EdgeDef(
            source=edge.source,
            target=edge.target,
            condition=getattr(edge, "data", None) if edge.conditional else None,
        )
        for edge in graph.edges
        if edge.source not in _SYNTHETIC and edge.target not in _SYNTHETIC
    ]
    return Topology(nodes=nodes, edges=edges)
