from langchain_openai import ChatOpenAI

from utils.langgraph.env_utils import SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY

DeepSeek_V3 = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)

Qwen3_30B_A3B_Instruct_2507 = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)



Qwen3_32B = ChatOpenAI(
    model="Qwen/Qwen3-32B",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)

Qwen3_Next_80B_A3B_Instruct = ChatOpenAI(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)


