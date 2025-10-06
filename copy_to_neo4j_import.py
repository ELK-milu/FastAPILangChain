"""
将 CSV 文件复制到 Neo4j import 目录的辅助脚本
"""
import shutil
from pathlib import Path

# 源文件目录
source_dir = Path(__file__).parent / "neo4j_import"

# 常见的 Neo4j import 目录位置
possible_neo4j_import_dirs = [
    Path.home() / "AppData/Roaming/Neo4j Desktop/Application/relate-data/dbmss",
    Path("C:/Program Files/Neo4j/import"),
    Path("C:/Neo4j/import"),
]

print("正在查找 Neo4j import 目录...")

# 查找 Neo4j import 目录
neo4j_import_dir = None
for base_dir in possible_neo4j_import_dirs:
    if base_dir.exists():
        if "dbmss" in str(base_dir):
            # 查找最新的 dbms 目录
            dbms_dirs = list(base_dir.glob("dbms-*/import"))
            if dbms_dirs:
                # 按修改时间排序，取最新的
                neo4j_import_dir = max(dbms_dirs, key=lambda p: p.stat().st_mtime)
                break
        elif base_dir.name == "import":
            neo4j_import_dir = base_dir
            break

if neo4j_import_dir:
    print(f"找到 Neo4j import 目录: {neo4j_import_dir}")

    # 复制所有 CSV 文件
    csv_files = list(source_dir.glob("*.csv"))
    if csv_files:
        for csv_file in csv_files:
            dest_file = neo4j_import_dir / csv_file.name
            shutil.copy2(csv_file, dest_file)
            print(f"  复制: {csv_file.name} -> {dest_file}")
        print(f"\n✓ 成功复制 {len(csv_files)} 个 CSV 文件到 Neo4j import 目录")
    else:
        print(f"✗ 在 {source_dir} 中没有找到 CSV 文件")
else:
    print("✗ 未找到 Neo4j import 目录")
    print("\n请手动将以下文件复制到 Neo4j 的 import 目录:")
    for csv_file in source_dir.glob("*.csv"):
        print(f"  - {csv_file}")
    print("\nNeo4j import 目录通常在:")
    print("  - C:\\Users\\<用户名>\\AppData\\Roaming\\Neo4j Desktop\\Application\\relate-data\\dbmss\\<dbms-id>\\import\\")
    print("  - 或者在 Neo4j 配置文件 (neo4j.conf) 中 server.directories.import 指定的位置")
