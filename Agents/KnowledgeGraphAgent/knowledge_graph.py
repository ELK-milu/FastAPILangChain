import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node
from utils.langgraph.Tools import request_user_approval
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
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    return user_goal_data


tools = [set_perceived_user_goal, request_user_approval]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node(tools)

system_prompt = SystemMessage(
    """你是一个知识图谱用例设计助手，帮助用户构思和定义知识图谱的应用场景。

工作流程：
1. 理解用户需求，分析他们想要构建的知识图谱类型
2. **必须先调用 request_user_approval 工具**获得人工审批，传入：
   - operation_description: 操作描述（例如："设定用户目标为BOM图表"）
3. 如果审批通过（approved=true），再调用 set_perceived_user_goal 工具设定用户目标
4. 如果审批被拒绝（approved=false），询问用户需要修改哪些方面

注意事项：
- 必须按顺序调用工具：先 request_user_approval，后 set_perceived_user_goal
- 审批工具的参数要详细、准确，便于人工判断
- 如果审批被拒绝，友好地询问用户反馈并重新提出方案
"""
)
call_chat_node = create_chat_node(chat_model, system_prompt)

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
config = {"configurable": {"thread_id": uuid.uuid4()}}
inputs = {"messages": [("user", "I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis.")]}
result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
    graph=graph,
    config=config,
    inputs=inputs,
    debug=False
)

print(agent_msgs)