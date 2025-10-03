from typing import Type, Optional

from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from utils.ChatNode import create_chat_node
from utils.ConditionNode import should_continue
from utils.OutputParser import agent_with_tool_stream_parser
from utils.ToolNode import create_tool_node
from Samples import model

class CalculatorInput(BaseModel):
    location: str = Field(description="the location to get the weather for")


class get_weather(BaseTool):
    name:str = "get_weather"
    description:str = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if any([city in location.lower() for city in ["sf", "san francisco"]]):
            return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
        else:
            return f"I am not sure what the weather is in {location}"

    async def _arun(
        self, location: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("get_weather does not support async")


tools = [get_weather()]

model = model.bind_tools(tools)
# 使用示例
generic_tool_node = create_tool_node(tools)
system_prompt = SystemMessage(
    "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
)
call_chat_node = create_chat_node(model,system_prompt)
# 定义循环结束条件

# 定义一个图
workflow = StateGraph(AgentState)

# 定义循环的两节点
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", generic_tool_node)

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


inputs = {"messages": [("user", "sf天气怎么样")]}
agent_with_tool_stream_parser(graph.stream(inputs, stream_mode="messages"))
