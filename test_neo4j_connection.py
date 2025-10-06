"""测试 Neo4j 连接"""
from neo4j import GraphDatabase
import sys

# Neo4j 连接配置
username = "neo4j"
password = "baidu123"
url = "bolt://localhost:7687"

try:
    print("正在连接到 Neo4j...")
    print(f"URL: {url}")
    print(f"用户名: {username}")

    # 使用原生驱动进行连接测试，设置连接超时
    driver = GraphDatabase.driver(
        url,
        auth=(username, password),
        connection_timeout=3.0,  # 3秒连接超时
        max_connection_lifetime=3600
    )

    # 验证连接
    driver.verify_connectivity()
    print("✓ 成功连接到 Neo4j")

    # 测试简单查询
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        record = result.single()
        print(f"✓ 测试查询成功: {record['test']}")

    driver.close()
    print("✓ 连接已关闭")

except Exception as e:
    print(f"✗ Neo4j 连接失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
