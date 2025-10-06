from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph

from Agents.KnowledgeGraphAgent import approved_construction_plan

#链接neo4j 账号密码
username = "neo4j"
password = "baidu123"
url = "bolt://localhost:7687"

graphdb = Neo4jGraph(
    url=url,
    username=username,
    password=password,
    refresh_schema=False  # 跳过 APOC 验证
)


#@tool(description="为节点标签和属性键创建唯一性约束。")
def create_uniqueness_constraint(
    label: str,
    unique_property_key: str,
):
    """
    为节点标签和属性键创建唯一性约束。
    唯一性约束确保具有相同标签和属性键的两个节点不会拥有相同的属性值。
    这将提升数据导入及后续查询的性能和完整性。

    Args:
        label: 需要创建约束的节点标签
        unique_property_key: 需要保持属性值唯一的属性键

    Return:
        dict:Neo4jdb返回的查询结果
    """
    # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint
    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
    FOR (n:`{label}`)
    REQUIRE n.`{unique_property_key}` IS UNIQUE"""
    results = graphdb.query(query)
    return results

#@tool(description="从CSV文件中批量加载节点。")
def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: list[str],
):
    """通过基于unique_column_name值进行合并的方式从CSV文件加载节点
    参数:
    file_path: 包含节点数据的CSV文件路径
    node_label: 要创建的节点标签
    unique_column_name: 用于唯一标识节点的列名

    Return:
        dict:Neo4jdb返回的查询结果
    """

    # load nodes from CSV file by merging on the unique_column_name value
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MERGE (n:$($label) {{ {unique_column_name} : row[$unique_column_name] }})
        FOREACH (k IN $properties | SET n[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    results = graphdb.query(query, {
        "source_file": source_file,
        "label": label,
        "unique_column_name": unique_column_name,
        "properties": properties
    })
    print(results)
    return results


def import_nodes(node_construction: dict):
    """Import nodes as defined by a node construction rule."""

    # create a uniqueness constraint for the unique_column
    uniqueness_result = create_uniqueness_constraint(
        node_construction["label"],
        node_construction["unique_column_name"]
    )
    print(uniqueness_result)
    #if (uniqueness_result["status"] == "error"):
        #return uniqueness_result

    # import nodes from csv
    load_nodes_result = load_nodes_from_csv(
        node_construction["source_file"],
        node_construction["label"],
        node_construction["unique_column_name"],
        node_construction["properties"]
    )

    return load_nodes_result


def import_relationships(relationship_construction: dict):
    """Import relationships as defined by a relationship construction rule."""

    # load nodes from CSV file by merging on the unique_column_name value
    from_node_column = relationship_construction["from_node_column"]
    to_node_column = relationship_construction["to_node_column"]
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MATCH (from_node:$($from_node_label) {{ {from_node_column} : row[$from_node_column] }}),
              (to_node:$($to_node_label) {{ {to_node_column} : row[$to_node_column] }} )
        MERGE (from_node)-[r:$($relationship_type)]->(to_node)
        FOREACH (k IN $properties | SET r[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    results = graphdb.query(query, {
        "source_file": relationship_construction["source_file"],
        "from_node_label": relationship_construction["from_node_label"],
        "from_node_column": relationship_construction["from_node_column"],
        "to_node_label": relationship_construction["to_node_label"],
        "to_node_column": relationship_construction["to_node_column"],
        "relationship_type": relationship_construction["relationship_type"],
        "properties": relationship_construction["properties"]
    })
    print(results)
    return results


def construct_domain_graph(construction_plan: dict):
    """Construct a domain graph according to a construction plan."""
    # first, import nodes
    node_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'node']
    for node_construction in node_constructions:
        import_nodes(node_construction)

    # second, import relationships
    relationship_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'relationship']
    for relationship_construction in relationship_constructions:
        import_relationships(relationship_construction)


construct_domain_graph(approved_construction_plan)