"""
测试继承 AgentState 的自定义 State
"""
import uuid
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
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
    print(f"[REDUCER] keep_latest: old={old}, new={new}")
    return new if new is not None else old


class SchemaProposalAgentState(AgentState):
    """继承 AgentState 的自定义 State"""
    test_num: Annotated[int, keep_latest]


@tool(description="测试工具")
@needs_state
def test_tool(user_input: str, state: SchemaProposalAgentState) -> dict:
    """测试工具"""
    print(f"\n[TEST TOOL] state keys: {list(state.keys())}")
    print(f"[TEST TOOL] 'test_num' in state: {'test_num' in state}")

    old_value = state.get("test_num", -999)
    print(f"[TEST TOOL] test_num 旧值: {old_value}")

    state["test_num"] = 100
    new_value = state["test_num"]
    print(f"[TEST TOOL] test_num 新值: {new_value}")

    return {
        "status": "success",
        "old_value": old_value,
        "new_value": new_value
    }


# 创建工具和模型
tools = [test_tool]
chat_model = DeepSeek_V3.bind_tools(tools)
tool_node = create_tool_node_with_state(tools)

system_prompt = "你是助手，使用 test_tool 处理用户输入"
call_chat_node = create_chat_node(chat_model, system_prompt)

# 构建工作流
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

# 编译并运行
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = {
    "messages": [HumanMessage(content="测试")],
    "test_num": 10
}

print("=" * 60)
print("测试继承 AgentState 的方案")
print("=" * 60)
print(f"输入: {inputs}\n")

try:
    for event in graph.stream(inputs, config):
        for node_name, node_output in event.items():
            print(f"\n[事件] 节点: {node_name}")
            if "test_num" in node_output:
                print(f"       test_num = {node_output['test_num']}")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
