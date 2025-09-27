import json

from langchain.agents import create_react_agent, AgentExecutor, initialize_agent, AgentType
from langchain.globals import set_debug
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


set_debug(True)

# graphdb = Neo4jGraph(url=NEO4J_URI,username=NEO4J_USERNAME,password=NEO4J_PASSWORD,refresh_schema=False)

agent_role_and_goal = """
你是知识图谱用例方面的专家。你的主要目标是帮助用户提出一个知识图谱用例。
"""

agent_conversational_hints = """
如果用户不确定该做什么，可以根据经典用例给出一些建议，例如：
- 涉及朋友、家人或职业关系的社交网络
- 包括供应商、客户和合作伙伴的物流网络
- 包括客户、产品和购买模式的推荐系统
- 对多个账户进行可疑交易模式的欺诈检测
- 包括电影、书籍或音乐的流行文化图谱
"""

agent_output_definition = """
用户目标有两个组成部分：
- 图表类型：最多3个单词描述图表，例如"社交网络"或"美国货运物流"。
- 描述：关于图表意图的几句话，例如"货物的动态路线和交付系统。"或"产品依赖关系和供应商替代品的分析。"
"""

agent_chain_of_thought_directions = """
仔细考虑并与用户协作：
1. 理解用户的目标，即一种带有描述的图表 
2. 根据需要提出澄清问题 
3. 当你认为理解了他们的目标时，使用'set_perceived_user_goal'工具记录你的感知 
4. 向用户展示感知的目标以供确认 
5. 如果用户同意，使用'approve_perceived_user_goal'工具来批准用户目标。这将把目标保存到状态中，作为'approved_user_goal'键下的值。
"""

# 修正：创建符合 ReAct agent 要求的 prompt 模板
complete_agent_instruction = f"""
{agent_role_and_goal}

{agent_conversational_hints}

{agent_output_definition}

{agent_chain_of_thought_directions}
"""

print(complete_agent_instruction)

# 修正：移除 ChatOpenAI 中的 prompt 参数
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-lmfljfvthonnvdwhtvgbzhvvbttzccciuqbgkplziczcogaa",
    # 移除了 prompt 参数
)
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool


class PerceivedToolContext(BaseModel):
    """Inputs to the wikipedia tool."""
    kind_of_graph: str = Field(
        description="用2-3个单词定义一个Graph,例如\"recent US patents\""
    )
    graph_description: str = Field(
        description="graph的单段描述，总结用户的意图"
    )


@tool("set_perceived_user_goal", args_schema=PerceivedToolContext)
def set_perceived_user_goal(kind_of_graph: str, graph_description: str):
    """设置用户感知的目标，包括Graph类型及其描述。
    Args:
        kind_of_graph: 用2-3个单词定义一个Graph,例如"recent US patents"
        graph_description: graph的单段描述，总结用户的意图
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    return user_goal_data


tools = [set_perceived_user_goal]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


import json


class NotApproved(Exception):
    """Custom exception."""


def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise NotApproved(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg


# 创建提示模板
template = complete_agent_instruction + '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(
    template
)


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

llm_with_tools = llm.bind_tools(tools)

# 创建 Agent
agent = (
        prompt
        | llm
)

# 创建 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
agent_executor.invoke({"input": "elk买了一把椅子，生成一个知识图谱"})
