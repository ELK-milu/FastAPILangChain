import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# Define graph state
class State(TypedDict):
    foo: str

# Define the nodes
def node_a(state: State):
    print("Called A")
    return {"foo": state["foo"] + "a"}

def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}

def pass_through(state: State) -> Literal["node_b", "node_c"]:
    value = random.choice(["node_b", "node_c"])
    print(f"Routing to: {value}")
    return value

builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
# 使用条件边
builder.add_conditional_edges("node_a", pass_through, {
    "node_b": "node_b",
    "node_c": "node_c"
})
builder.add_edge("node_b", END)
builder.add_edge("node_c", END)

graph = builder.compile()

graph.invoke({"foo": ""})
