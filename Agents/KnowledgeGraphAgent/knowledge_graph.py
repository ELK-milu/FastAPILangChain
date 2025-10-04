import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command

from Agents.KnowledgeGraphAgent import model
from utils.ChatNode import create_chat_node
from utils.ConditionNode import should_continue
from utils.HumanApproval import HumanApproval
from utils.OutputParser import agent_with_tool_stream_parser, run_workflow_with_approval, run_workflow_with_approval_streaming
from utils.ToolNode import create_tool_node, create_tool_node_with_approval

PERCEIVED_USER_GOAL = "perceived_user_goal"

class GlobalState(AgentState):
    user_goal_data: dict

@tool(description="设定感知用户目标，包括graph类型及其描述。")
def set_perceived_user_goal(kind_of_graph: str, graph_description: str):
    """设定感知用户的目标，包括图表类型及其描述。

    Args:
        kind_of_graph: 用2-3个词定义图表类型，例如"recent US patents"
        graph_description: 一段描述图表内容的段落，概括用户意图
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    return user_goal_data

def approval_node(state: AgentState):
    return state

def rejected_node(state: AgentState):
    return state


APPROVED_USER_GOAL = "approved_user_goal"

tools = [set_perceived_user_goal]

chat_model = model.bind_tools(
    tools
)
tool_node = create_tool_node_with_approval(tools,"approval_node","rejected_node")

system_prompt = SystemMessage(
    "帮助用户构思知识图用例"
)
call_chat_node = create_chat_node(chat_model, system_prompt)

workflow = StateGraph(GlobalState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.add_node("approval_node", approval_node)
workflow.add_node("rejected_node", rejected_node)
workflow.set_entry_point("agent")
# 添加条件边
workflow.add_conditional_edges(
    # 起点：agent节点
    "agent",
    # 调用agent后的hook函数
    should_continue,
    # 根据hook函数返回的结果进行节点调用映射
    # 若hook返回continue则调用tools节点，若为end则调用END节点
    # END节点是一个特殊的节点，就是workflow的结束
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)
workflow.add_edge("approval_node", "agent")  # 批准后回到 agent
workflow.add_edge("rejected_node", END)  # 拒绝后结束对话，或者也可以回到 agent
checkpointer = InMemorySaver()

# 编译workflow为一个graph对象
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}
inputs = {"messages": [("user", "I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis.")]}
result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
    graph=graph,
    config=config,
    inputs=inputs,
    debug=False
)

print(agent_msgs)