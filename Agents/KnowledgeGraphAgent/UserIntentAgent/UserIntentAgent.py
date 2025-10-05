import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool

from Agents.KnowledgeGraphAgent.UserIntentAgent import complete_agent_instruction
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node
from utils.langgraph.Tools import get_approved_user_goal
from utils.models import DeepSeek_V3

PERCEIVED_USER_GOAL = "perceived_user_goal"

class GlobalState(AgentState):
    user_goal_data: dict

@tool(description="设定感知用户目标，包括graph类型及其描述。")
def set_perceived_user_goal(kind_of_graph: str, graph_description: str):
    """设定感知用户的目标，包括图表类型及其描述。

    Args:
        kind_of_graph: 用2-3个词定义图表类型，例如"recent US patents"
        graph_description: 一段描述图表内容的段落，概括用户意图
    Returns:
        user_goal_data: 包含kind_of_graph和graph_description的字典
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    return user_goal_data


tools = [set_perceived_user_goal, get_approved_user_goal]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node(tools)

call_chat_node = create_chat_node(chat_model, complete_agent_instruction)

workflow = StateGraph(GlobalState)
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

if __name__ == "__main__":

    config = {"configurable": {"thread_id": uuid.uuid4()}}
    inputs = {"messages": [("user", "I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis.")]}
    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=graph,
        config=config,
        inputs=inputs,
        debug=False
    )

