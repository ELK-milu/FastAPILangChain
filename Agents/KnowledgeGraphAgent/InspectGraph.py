# extract a list of the relationship construction rules
from Agents.KnowledgeGraphAgent import approved_construction_plan, graphdb

relationship_constructions = [
    value for value in approved_construction_plan.values()
    if value.get("construction_type") == "relationship"
]

# a fancy cypher query which to show one instance of each construction rule

# turn the list of rules into multiple single rules
unwind_list = "UNWIND $relationship_constructions AS construction"

# match a single path for a given construction.relationship_type
# return only the labels and types from the 3 parts of the path
match_one_path = """
    MATCH (from)-[r:$(construction.relationship_type)]->(to)
    RETURN labels(from) AS fromNode, type(r) AS relationship, labels(to) AS toNode
    LIMIT 1
"""
match_in_subquery = f"""
CALL (construction) {{
{match_one_path}
}}
"""

cypher = f"""
{unwind_list}
{match_in_subquery}
RETURN fromNode, relationship, toNode
"""

print(cypher)

print("\n---")

graphdb.query(cypher, {
    "relationship_constructions": relationship_constructions
})