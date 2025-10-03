# 定义工具调用节点
import json

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.chat_agent_executor import AgentState


def create_tool_node(tools):
    """
    创建一个通用的工具调用节点
    :param tools: 工具列表
    """
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    return tool_node
