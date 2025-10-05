"""
测试 test_num 字段的全局传递
"""
import uuid
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.managed import RemainingSteps
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.models import DeepSeek_V3


def keep_latest(old, new):
    """保留最新值的 Reducer：只在提供新值时更新"""
    return new if new is not None else old


class TestState(TypedDict):
    """测试 State 定义"""
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    test_num: Annotated[int, keep_latest]


@tool(description="测试工具：读取和修改 test_num")
@needs_state
def test_tool(user_input: str, state: TestState) -> dict:
    """测试工具"""
    print(f"\n[TEST] 工具中的 state keys: {list(state.keys())}")
    print(f"[TEST] 'test_num' in state: {'test_num' in state}")

    old_value = state.get("test_num", -999)
    print(f"[TEST] test_num 当前值: {old_value}")

    # 修改 test_num
    state["test_num"] = 100
    new_value = state["test_num"]
    print(f"[TEST] test_num 新值: {new_value}")

    return {
        "status": "success",
        "old_value": old_value,
        "new_value": new_value,
        "message": f"复读: {user_input}"
    }


# 创建工具和模型
tools = [test_tool]
chat_model = DeepSeek_V3.bind_tools(tools)
tool_node = create_tool_node_with_state(tools)

system_prompt = "你是一个助手。使用 test_tool 工具来处理用户输入。"
call_chat_node = create_chat_node(chat_model, system_prompt)

# 构建工作流
workflow = StateGraph(TestState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

# 编译并运行
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}

# 初始化 inputs（包含 test_num）
inputs = {
    "messages": [HumanMessage(content="测试消息")],
    "test_num": 10  # 初始值设为 10
}

print("=" * 60)
print("开始测试 test_num 字段传递")
print("=" * 60)
print(f"\n初始化 inputs: {inputs}\n")

# 简单运行（不使用 streaming）
try:
    for event in graph.stream(inputs, config):
        for node_name, node_output in event.items():
            print(f"\n[NODE: {node_name}]")
            if "test_num" in node_output:
                print(f"  test_num = {node_output['test_num']}")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    print(f"  message: {type(msg).__name__}")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
