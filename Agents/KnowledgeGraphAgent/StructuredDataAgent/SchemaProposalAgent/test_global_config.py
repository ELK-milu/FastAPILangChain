import uuid
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.managed import RemainingSteps
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

import sys

from langgraph.prebuilt.chat_agent_executor import AgentState

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from Agents.KnowledgeGraphAgent.StructuredDataAgent.SchemaProposalAgent import proposal_agent_instruction
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state, create_tool_node
from utils.models import DeepSeek_V3
from utils.neo4j import get_neo4j_import_dir

SEARCH_RESULTS = "search_results"


def keep_latest(old, new):
    """保留最新值的 Reducer：只在提供新值时更新"""
    return new if new is not None else old


class SchemaProposalAgentState(AgentState):
    # 不再在 State 中存储全局配置，改用 config["configurable"]
    pass


@tool(description="复读用户的话")
def test_node(user_input: str, config: RunnableConfig) -> dict:
    """
    复读用户的话，并访问全局配置。

    参数:
        user_input: 用户的输入
        config: 运行时配置（LangGraph 自动注入）

    返回:
        dict: 包含内容元数据的字典。
              包含 'status' 键（'success' 或 'error'）。
              如果是 'success'，包含 'user_input' 键和全局配置信息。
              如果是 'error'，包含 'error_message' 键。
    """
    # 从 config["configurable"] 中读取全局配置
    configurable = config.get("configurable", {})
    test_num = configurable.get("test_num", 0)
    user_id = configurable.get("user_id", "unknown")

    print(f"\n[DEBUG] 从 config 读取全局配置:")
    print(f"[DEBUG] test_num: {test_num} (类型: {type(test_num)})")
    print(f"[DEBUG] user_id: {user_id}")
    print(f"[DEBUG] 完整 configurable: {configurable}")

    return {
        "status": "success",
        "user_input": user_input,
        "test_num": test_num,
        "user_id": user_id,
        "message": f"复读: {user_input}, test_num={test_num}, user={user_id}"
    }

def print_node(state: SchemaProposalAgentState, config: RunnableConfig):
    """打印节点：从 config 读取全局配置"""
    configurable = config.get("configurable", {})
    test_num = configurable.get("test_num", 0)

    # 演示：全局配置是不可变的，这里只是计算新值但不修改原配置
    new_value = test_num + 50
    print(f"\n[DEBUG] print_node - 原始 test_num: {test_num}")
    print(f"[DEBUG] print_node - 计算后的值: {new_value}")
    print(f"[DEBUG] 注意：全局配置在整个 workflow 中保持不变")

    return state

tools = [test_node]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node(tools)

system_prompt = """
    你是一个助手，你需要完成一个任务。
    你需要使用test_node工具来完成复读用户的输入。
    """

call_chat_node = create_chat_node(chat_model, system_prompt)

workflow = StateGraph(SchemaProposalAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.add_node("print_node", print_node)
workflow.set_entry_point("agent")

# 添加条件边：agent 决定是否调用工具
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # 调用工具
        "end": "print_node",           # 结束对话
    },
)

# 工具执行后返回 agent 继续处理
workflow.add_edge("tools", "agent")
workflow.add_edge("print_node", END)

checkpointer = InMemorySaver()

# 编译 workflow
graph = workflow.compile(checkpointer)

# 使用 config["configurable"] 传递全局配置（推荐方案）
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "test_num": 10,          # 全局配置：测试数字
        "user_id": "user_123",   # 全局配置：用户 ID
        "neo4j_import_dir": get_neo4j_import_dir()  # 全局配置：Neo4j 导入目录
    }
}

# inputs 只包含业务状态数据
inputs = {
    "messages": [("user", "北京市朝阳区")]
}

print("\n=== 使用 config['configurable'] 传递全局配置 ===")
print(f"全局配置: {config['configurable']}")
print(f"业务输入: {inputs}\n")

result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
    graph=graph,
    config=config,
    inputs=inputs,
    debug=False
)
