# 工具列表
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
class PerceivedToolContext(BaseModel):
    """Inputs to the wikipedia tool."""
    kind_of_graph: str = Field(
        description="用2-3个单词定义一个Graph,例如\"recent US patents\""
    )
    graph_description: str = Field(
        description="graph的单段描述，总结用户的意图"
    )


@tool("set_perceived_user_goal")
def set_perceived_user_goal(kind_of_graph: str, graph_description: str):
    """设置用户感知的目标，包括Graph类型及其描述。
    Args:
        kind_of_graph: 用2-3个单词定义一个Graph,例如"recent US patents"
        graph_description: graph的单段描述，总结用户的意图
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    return user_goal_data

tools = [
    set_perceived_user_goal
]

react_prompt = ChatPromptTemplate.from_template(
    agent_chain_of_thought_directions + """
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
    Thought:{agent_scratchpad}
"""
)

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-lmfljfvthonnvdwhtvgbzhvvbttzccciuqbgkplziczcogaa",
)

# 创建ReAct代理
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# 创建代理执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细执行过程
    # early_stopping_method="generate",  # 提前停止策略
    handle_parsing_errors=True  # 处理解析错误
)

agent_executor.invoke({"input": "我想创建一个关于近期美国专利的知识图谱"})