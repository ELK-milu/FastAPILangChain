from pathlib import Path

from langchain_core.tools import tool

from utils.langgraph.env_utils import NEO4J_IMPORT_DIR

DEFAULT_NEO4J_IMPORT_DIR = "E:\\DockerImages\\neo4j\\import"
def get_neo4j_import_dir() -> Path:
    """
    获取 Neo4j 导入目录路径

    优先级：
    1. 环境变量 NEO4J_IMPORT_DIR
    2. Neo4j 配置文件中的 dbms.directories.import
    3. 默认路径 ./neo4j_import

    Returns:
        Path: Neo4j 导入目录的 Path 对象
    """
    # 1. 尝试从环境变量获取
    import_dir_env = NEO4J_IMPORT_DIR
    if import_dir_env:
        import_dir = Path(import_dir_env)
    else:
        # 2. 使用默认路径
        import_dir = Path(DEFAULT_NEO4J_IMPORT_DIR)

    # 如果目录不存在，创建它
    if not import_dir.exists():
        import_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建 Neo4j 导入目录: {import_dir.resolve()}")

    return import_dir




if __name__ == "__main__":
    print(get_neo4j_import_dir())
