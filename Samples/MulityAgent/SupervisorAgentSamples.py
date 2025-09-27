import random
from typing import Type, Optional
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from utils.env_utils import SILICONFLOW_BASE_URL,SILICONFLOW_API_KEY

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)


class SearchInput(BaseModel):
    city: str = Field(description="需要搜索的城市")


class web_search(BaseTool):
    name:str = "web_search"
    description:str = "useful for when you need to web search"
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self, city: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return f"{city}的GDP是{random.randint(1000,5000)}亿美元"

    async def _arun(
        self, city: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("get_weather does not support async")

from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    model=model,
    tools=[web_search()],
    prompt=(
        "你是一个研究助手。\n\n"
        "指令：\n"
        "- 仅协助完成研究相关任务，不要进行任何数学计算\n"
        "- 完成任务后，直接向主管汇报结果\n"
        "- 仅回复你的工作结果，不要包含任何其他文本"
    ),
    name="research_agent",
)

from langchain_core.messages import convert_to_messages

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

'''
for chunk in research_agent.stream(
    {"messages": [{"role": "user", "content": "北京的GDP是？"}]}
):
    pretty_print_messages(chunk)
'''

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply, divide],
    prompt=(
        "你是一个数学助手。\n\n"
        "指令：\n"
        "- 仅协助完成数学相关任务\n"
        "- 完成任务后，直接向主管汇报结果\n"
        "- 仅回复你的工作结果，不要包含任何其他文本"
    ),
    name="math_agent",
)
'''
for chunk in math_agent.stream(
    {"messages": [{"role": "user", "content": "1+5=？"}]}
):
    pretty_print_messages(chunk)
'''

from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model(model_provider="openai", model="deepseek-ai/DeepSeek-V3",base_url=SILICONFLOW_BASE_URL,api_key=SILICONFLOW_API_KEY),
    agents=[research_agent, math_agent],
    prompt=(
        "你是一名主管，管理两个助手：\n"
        "- 研究助手：将研究相关任务分配给这个助手\n"
        "- 数学助手：将数学相关任务分配给这个助手\n"
        "每次只给一个助手分配工作，不要同时调用两个助手。\n"
        "不要自己完成任何工作。"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "找出北京和上海的GDP，并告诉我二者相加之和",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]