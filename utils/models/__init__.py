from langchain_openai import ChatOpenAI

from utils.langgraph.env_utils import SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY

DeepSeek_V3 = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)
