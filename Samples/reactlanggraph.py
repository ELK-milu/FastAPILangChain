from typing import Type, Optional

from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt.chat_agent_executor import AgentState
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from pydantic import BaseModel, Field

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-lmfljfvthonnvdwhtvgbzhvvbttzccciuqbgkplziczcogaa",
)

'''
@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"I am not sure what the weather is in {location}"
'''
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


tools_by_name = {tool.name: tool for tool in tools}


# 定义工具调用节点
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


# 定义模型调用节点
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # 定义prompt提示词
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # 需要返回一个messages列表
    return {"messages": [response]}


# 定义循环结束条件
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果无工具调用则结束
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"



# 定义一个图
workflow = StateGraph(AgentState)

# 定义循环的两节点
workflow.add_node("agent", call_model)
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


agent_responses = []
tool_responses = []


# 对消息对象进行流式输出
def print_stream(stream):
    for chunk in stream:
        message_chunk, metadata = chunk
        node_name = metadata.get('langgraph_node', 'unknown')
        print(message_chunk)

        if hasattr(message_chunk, 'content') and message_chunk.content:
            if node_name == 'agent':
                agent_responses.append(message_chunk.content)
                #print(f"{message_chunk.content}", end="", flush=True)
            elif node_name == 'tools':
                tool_responses.append(message_chunk.content)
                #print(f"tools:{message_chunk.content}")

inputs = {"messages": [("user", "sf天气怎么样")]}
print_stream(graph.stream(inputs, stream_mode="messages"))