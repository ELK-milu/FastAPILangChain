from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from Agents.KnowledgeGraphAgent import model,complete_agent_instruction


class KnowledgeGraphState(TypedDict):
    """定义知识图谱代理的状态"""
    messages: List[BaseMessage]
    perceived_user_goal: Optional[Dict[str, str]]
    approved_user_goal: Optional[Dict[str, str]]


# 全局状态变量，用于在工具调用之间保持状态
_current_state: Optional[KnowledgeGraphState] = None


@tool
def set_perceived_user_goal(kind_of_graph: str, graph_description: str) -> str:
    """设置感知的用户目标，包括图谱类型和描述。

    Args:
        kind_of_graph: 2-3个词定义的图谱类型，例如"近期美国专利"
        graph_description: 对图谱的单段描述，总结用户意图
    """
    global _current_state
    if _current_state is None:
        return "错误：状态未初始化"

    user_goal_data = {
        "kind_of_graph": kind_of_graph,
        "graph_description": graph_description
    }
    _current_state["perceived_user_goal"] = user_goal_data

    return f"已设置感知的用户目标：{json.dumps(user_goal_data, ensure_ascii=False)}"


@tool
def approve_perceived_user_goal() -> str:
    """在用户批准后，将感知的用户目标记录为已批准的用户目标。

    仅当用户明确批准了感知的用户目标时才调用此工具。
    """
    global _current_state
    if _current_state is None:
        return "错误：状态未初始化"

    if _current_state.get("perceived_user_goal") is None:
        return "错误：未设置感知的用户目标。请先设置感知的用户目标，或如果不确定请询问澄清问题。"

    _current_state["approved_user_goal"] = _current_state["perceived_user_goal"]

    return f"已批准用户目标：{json.dumps(_current_state['approved_user_goal'], ensure_ascii=False)}"


# 创建工具列表
knowledge_graph_tools = [set_perceived_user_goal, approve_perceived_user_goal]

# 绑定工具到模型
model_with_tools = model.bind_tools(knowledge_graph_tools)

# 创建工具节点
tool_node = ToolNode(knowledge_graph_tools)


def agent_node(state: KnowledgeGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """代理节点：处理用户输入并生成响应"""
    global _current_state
    _current_state = state

    # 添加系统提示
    system_message = SystemMessage(content=complete_agent_instruction)

    # 构建消息列表
    messages = [system_message] + state["messages"]

    # 调用模型
    response = model_with_tools.invoke(messages, config)

    return {"messages": [response]}


def should_continue(state: KnowledgeGraphState) -> str:
    """决定是否继续工具调用还是结束"""
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]

    # 如果最后一条消息有工具调用，继续执行工具
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    else:
        return "end"


def tool_node_wrapper(state: KnowledgeGraphState) -> Dict[str, Any]:
    """工具节点包装器：更新全局状态并执行工具"""
    global _current_state
    _current_state = state

    # 调用工具节点
    result = tool_node.invoke(state)

    # 更新状态
    if _current_state:
        result["perceived_user_goal"] = _current_state.get("perceived_user_goal")
        result["approved_user_goal"] = _current_state.get("approved_user_goal")

    return result


# 创建状态图
workflow = StateGraph(KnowledgeGraphState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node_wrapper)

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

# 添加从工具回到代理的边
workflow.add_edge("tools", "agent")

# 编译图
user_intent_agent = workflow.compile()

print("Knowledge Graph Agent 已使用 LangGraph StateGraph 创建完成。")

user_intent_agent.invoke({"input": "我想创建一个关于近期美国专利的知识图谱"})