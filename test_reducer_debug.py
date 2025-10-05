"""
调试 Reducer 行为
"""
import uuid
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.managed import RemainingSteps
from typing import Annotated, TypedDict
from langgraph.graph import add_messages


def keep_latest(old, new):
    """保留最新值的 Reducer"""
    print(f"[REDUCER] keep_latest called: old={old}, new={new}, returning={new if new is not None else old}")
    return new if new is not None else old


class TestState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    test_num: Annotated[int, keep_latest]


def first_node(state: TestState):
    print(f"\n[FIRST_NODE] 接收 state keys: {list(state.keys())}")
    print(f"[FIRST_NODE] test_num in state: {'test_num' in state}")
    print(f"[FIRST_NODE] test_num value: {state.get('test_num', 'MISSING')}")

    # 返回 messages 和 test_num
    result = {
        "messages": [HumanMessage(content="from first_node")],
        "test_num": state.get("test_num", -1)  # 显式返回
    }
    print(f"[FIRST_NODE] 返回: {result}")
    return result


# 构建工作流
workflow = StateGraph(TestState)
workflow.add_node("first", first_node)
workflow.set_entry_point("first")
workflow.add_edge("first", END)

# 编译并运行
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}

# 初始化 inputs
inputs = {
    "messages": [HumanMessage(content="initial")],
    "test_num": 10
}

print("=" * 60)
print("调试 Reducer 行为")
print("=" * 60)
print(f"输入: {inputs}\n")

for event in graph.stream(inputs, config):
    print(f"\n[EVENT] {event}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
