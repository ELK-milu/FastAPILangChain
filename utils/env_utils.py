import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
SILICONFLOW_BASE_URL=os.environ.get("SILICONFLOW_BASE_URL")
SILICONFLOW_API_KEY=os.environ.get("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY=os.environ.get("DEEPSEEK_API_KEY")
NEO4J_URI=os.environ.get("NEO4J_URI")
NEO4J_USERNAME= os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD= os.environ.get("NEO4J_PASSWORD")
