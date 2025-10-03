from langchain_openai import ChatOpenAI

from utils.env_utils import SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
)