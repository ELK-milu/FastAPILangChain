"""
测试 SchemaProposalAgent 的 test_num 传递
"""
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Annotated

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.models import DeepSeek_V3


def keep_latest(old, new):
    """保留最新值的 Reducer"""
    return new if new is not None else old


class SchemaProposalAgentState(AgentState):
    test_num: Annotated[int, keep_latest]


@tool(description="复读用户的话")
@needs_state
def test_node(user_input: str, state: SchemaProposalAgentState) -> dict:
    """复读用户的话"""
    print(f"\n[工具执行] state keys: {list(state.keys())}")
    print(f"[工具执行] test_num = {state.get('test_num', 'MISSING')}")

    old_value = state.get("test_num", -999)
    state["test_num"] = 100

    return {
        "status": "success",
        "user_input": user_input,
        "old_value": old_value,
        "new_value": 100,
        "message": f"复读: {user_input}"
    }


tools = [test_node]
chat_model = DeepSeek_V3.bind_tools(tools)
tool_node = create_tool_node_with_state(tools)

system_prompt = "你是助手，使用 test_node 工具处理用户输入"
call_chat_node = create_chat_node(chat_model, system_prompt)

workflow = StateGraph(SchemaProposalAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
workflow.add_edge("tools", "agent")

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = {
    "messages": [HumanMessage(content="北京市朝阳区")],
    "test_num": 10
}

print("=" * 60)
print("测试 SchemaProposalAgent test_num 传递")
print("=" * 60)
print(f"初始化: test_num={inputs['test_num']}\n")

for event in graph.stream(inputs, config):
    for node_name, node_output in event.items():
        print(f"\n[节点: {node_name}]")
        if "test_num" in node_output:
            print(f"  test_num = {node_output['test_num']}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
