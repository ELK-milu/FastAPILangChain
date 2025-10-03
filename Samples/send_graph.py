import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.types import Send
from langgraph.graph import END, StateGraph, START


# graph的OverallState
class OverallState(TypedDict):
    topic: str
    subjects: list
    # 使用operator.add来将返回的消息组合为list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str

# 模拟根据动物主题生成的subjects节点
def generate_topics(state: OverallState):
    # Simulate a LLM.
    return {"subjects": ["lions", "elephants", "penguins"]}


# 模拟三种动物分别返回的笑话
def generate_joke(state: JokeState):
    # Simulate a LLM.
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": (
            "Why don’t penguins like talking to strangers at parties? "
            "Because they find it hard to break the ice."
        ),
    }
    return {"jokes": [joke_map[state["subject"]]]}


# 使用Send指令来控制条件边
def continue_to_jokes(state: OverallState):
    # 遍历OverallState的所有动物主题并生成Send指令调用generate_joke节点
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# 模拟最佳笑话，是企鹅的笑话
def best_joke(state: OverallState):
    return {"best_selected_joke": "penguins"}


builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)
builder.add_edge(START, "generate_topics")
# 在generate_topics节点后添加了条件边continue_to_jokes，调用后会根据send指令调用generate_joke节点
# path_map其实没有必要加，通过send指令直接跳转了。只不过加上去能让绘制的graph更完整
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)
graph = builder.compile()

for step in graph.stream({"topic": "animals"}):
    print(step)