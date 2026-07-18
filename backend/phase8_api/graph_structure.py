# Builds the static node/edge structure the frontend renders as the
# decision diagram. Node names come from the real compiled graphs wherever
# possible so the diagram never drifts from what the backend actually runs.
from phase5_critic.graph import graph as card_optimizer_graph
from phase7_human_loop.graph import graph as approval_graph

# Hardcoded rather than introspected live: the subscription hunter's
# create_agent() constructs a ChatGroq eagerly, so calling .get_graph() on
# it would require a live GROQ_API_KEY just to describe the graph's shape.
# This exact shape was confirmed by direct inspection, on
# langchain 1.3.14 / langgraph 1.2.9.
_SUBSCRIPTION_HUNTER_NODES = ["model", "tools"]
_SUBSCRIPTION_HUNTER_EDGES = [("model", "tools"), ("tools", "model")]


def build_graph_structure() -> dict:
    outer_edges = {(edge.source, edge.target) for edge in approval_graph.get_graph().edges}
    outer_nodes = {node for pair in outer_edges for node in pair if node not in ("__start__", "__end__")}

    card_edges = {(edge.source, edge.target) for edge in card_optimizer_graph.get_graph().edges}
    card_nodes = {node for pair in card_edges for node in pair if node not in ("__start__", "__end__")}

    nodes = (
        [{"id": name, "graph": "outer"} for name in sorted(outer_nodes)]
        + [{"id": name, "graph": "card_optimizer"} for name in sorted(card_nodes)]
        + [{"id": name, "graph": "subscription_hunter"} for name in _SUBSCRIPTION_HUNTER_NODES]
    )
    edges = (
        [{"source": s, "target": t, "graph": "outer"} for s, t in sorted(outer_edges) if s != "__start__" and t != "__end__"]
        + [{"source": s, "target": t, "graph": "card_optimizer"} for s, t in sorted(card_edges) if s != "__start__" and t != "__end__"]
        + [{"source": s, "target": t, "graph": "subscription_hunter"} for s, t in _SUBSCRIPTION_HUNTER_EDGES]
        + [
            {"source": "dispatch_node", "target": "reason", "graph": "fan_out", "label": "card_optimizer or both"},
            {"source": "dispatch_node", "target": "model", "graph": "fan_out", "label": "subscription_hunter or both"},
        ]
    )
    return {"nodes": nodes, "edges": edges}
