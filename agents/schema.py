from haystack.components.agents.state import State

schema = {
    "query": {"type": str},
    "documentation": {"type": list},
}

state = State(
    schema=schema
)
