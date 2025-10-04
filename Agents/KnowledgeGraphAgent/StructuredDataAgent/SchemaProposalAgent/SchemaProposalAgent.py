import uuid
from pathlib import Path
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from Agents.KnowledgeGraphAgent.StructuredDataAgent.SchemaProposalAgent import proposal_agent_instruction
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node_with_state
from utils.models import DeepSeek_V3
from utils.neo4j import get_neo4j_import_dir

SEARCH_RESULTS = "search_results"

class SchemaProposalAgentState(AgentState):
    _propose_node_construction_plan :dict = {"test":"usertest"}


@tool(description="复读用户的话")
def test_node(user_input: str, state: SchemaProposalAgentState) -> dict:
    """
    复读用户的话，并访问/修改 state。

    参数:
        user_input: 用户的输入
        state: Agent 状态（由 create_tool_node_with_state 自动注入）

    返回:
        dict: 包含内容元数据的字典。
              包含 'status' 键（'success' 或 'error'）。
              如果是 'success'，包含 'user_input' 键和 state 信息。
              如果是 'error'，包含 'error_message' 键。
              'error_message' 可能包含关于如何处理错误的说明。
    """
    # 读取当前的 construction plan
    construction_plan = state.get("_propose_node_construction_plan", {})
    print(f"当前 construction plan: {construction_plan}")

    # 修改 state
    state["_propose_node_construction_plan"] = {"process": user_input}
    new_construction_plan = state["_propose_node_construction_plan"]
    print(f"更新后 construction plan: {new_construction_plan}")

    return {
        "status": "success",
        "user_input": user_input,
        "old_plan": construction_plan,
        "new_plan": new_construction_plan,
        "message": f"复读: {user_input}"
    }

# 标记 test_node 需要 state
test_node.needs_state = True

tools = [test_node]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node_with_state(tools)

system_prompt = """
    你是一个助手，你需要完成一个任务。
    你需要使用test_node工具来完成复读用户的输入。
    """

call_chat_node = create_chat_node(chat_model, system_prompt)

workflow = StateGraph(SchemaProposalAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

# 添加条件边：agent 决定是否调用工具
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # 调用工具
        "end": END,           # 结束对话
    },
)

# 工具执行后返回 agent 继续处理
workflow.add_edge("tools", "agent")

checkpointer = InMemorySaver()

# 编译 workflow
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}
inputs = {"messages": [("user", "北京市朝阳区")]}
result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
    graph=graph,
    config=config,
    inputs=inputs,
    debug=False
)
