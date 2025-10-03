from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage

from Agents.KnowledgeGraphAgent import model
from utils.ChatNode import create_chat_node
from utils.ConditionNode import should_continue
from utils.OutputParser import agent_with_tool_stream_parser
from utils.ToolNode import create_tool_node


@tool(description="天气搜索助手,当你需要查找某个城市的天气时调用")
def weather_search(city: str):
    """
    天气搜索助手,用于查找某个城市的天气
    Args:
        city (str): 城市名
    Returns:
        str: 天气情况
    """
    print("----")
    print(f"正在搜索: {city}")
    print("----")
    return "晴天!"

tools = [weather_search]

chatmodel = model.bind_tools(
    tools
)
tool_node = create_tool_node(tools)
system_prompt = SystemMessage(
    "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
)
call_chat_node = create_chat_node(chatmodel,system_prompt)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
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

# 为tools添加回到agent的循环
workflow.add_edge("tools", "agent")

# 编译workflow为一个graph对象
graph = workflow.compile()

inputs = {"messages": [("user", "北京天气怎么样")]}
agent_with_tool_stream_parser(graph.stream(inputs, stream_mode="messages"),debug=True)
