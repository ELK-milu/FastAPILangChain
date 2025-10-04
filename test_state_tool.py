"""
简化测试脚本：验证 create_tool_node_with_state 和 @needs_state 装饰器
"""
import sys
import uuid
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.models import DeepSeek_V3

# 定义自定义 State
class TestAgentState(AgentState):
    test_data: dict = {"initial": "value"}

# 定义需要 state 的工具
@tool(description="测试工具：访问和修改 state")
@needs_state
def test_tool(user_input: str, state: TestAgentState) -> dict:
    """
    测试工具，读取和修改 state。

    参数:
        user_input: 用户输入
        state: Agent 状态（自动注入）

    返回:
        dict: 工具执行结果
    """
    print(f"\n[工具执行] 收到用户输入: {user_input}")

    # 读取当前 state
    old_data = state.get("test_data", {})
    print(f"[工具执行] 当前 state: {old_data}")

    # 修改 state
    state["test_data"] = {"user_input": user_input, "processed": True}
    new_data = state["test_data"]
    print(f"[工具执行] 更新后 state: {new_data}")

    return {
        "status": "success",
        "user_input": user_input,
        "old_data": old_data,
        "new_data": new_data,
        "message": f"已处理: {user_input}"
    }

# 创建工具列表
tools = [test_tool]

# 绑定工具到模型
chat_model = DeepSeek_V3.bind_tools(tools)

# 创建工具节点
tool_node = create_tool_node_with_state(tools)

# 系统提示
system_prompt = """
你是一个测试助手。
请使用 test_tool 工具来处理用户的输入。
"""

# 创建 chat 节点
call_chat_node = create_chat_node(chat_model, system_prompt)

# 构建工作流
workflow = StateGraph(TestAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# 工具执行后返回 agent
workflow.add_edge("tools", "agent")

# 编译工作流
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer)

print("=" * 60)
print("🚀 开始测试 create_tool_node_with_state 功能")
print("=" * 60)

# 执行测试
config = {"configurable": {"thread_id": uuid.uuid4()}}
inputs = {"messages": [("user", "测试输入123")]}

try:
    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=graph,
        config=config,
        inputs=inputs,
        debug=True
    )

    print("\n" + "=" * 60)
    print("✅ 测试成功！")
    print("=" * 60)
    print(f"\n最终结果: {result}")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 测试失败！")
    print("=" * 60)
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
