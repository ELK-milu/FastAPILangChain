from langchain_openai import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from Agents.KnowledgeGraphAgent import graphdb
from utils.langgraph.env_utils import SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# use the same driver set up by neo4j_for_adk.py
neo4j_driver = graphdb._driver
llm_for_neo4j = OpenAILLM(
    model_name="deepseek-ai/DeepSeek-V3",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY
)

embedder = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY
)


approved_entities = ['Product', 'Issue', 'Feature', 'Location']

# approved fact types from the `relevant_fact_agent` of Lesson 7
approved_fact_types = {'has_issue': {'subject_label': 'Product', 'predicate_label': 'has_issue', 'object_label': 'Issue'}, 'includes_feature': {'subject_label': 'Product', 'predicate_label': 'includes_feature', 'object_label': 'Feature'}, 'used_in_location': {'subject_label': 'Product', 'predicate_label': 'used_in_location', 'object_label': 'Location'}}


# per-chunk entity extraction prompt, with context
def contextualize_er_extraction_prompt(context: str) -> str:
    """Creates a prompt with pre-amble file content for context during entity+relationship extraction.
    The context is concatenated into the string, which later will be used as a template
    for values like {schema} and {text}.
    """
    general_instructions = """
    You are a top-tier algorithm designed for extracting
    information in structured formats to build a knowledge graph.

    Extract the entities (nodes) and specify their type from the following text.
    Also extract the relationships between these nodes.

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
    "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

    Use only the following node and relationship types (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Make sure you adhere to the following rules to produce valid JSON objects:
    - Do not return any additional information other than the JSON in it.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - Property names must be enclosed in double quotes
    """

    context_goes_here = f"""
    Consider the following context to help identify entities and relationships:
    <context>
    {context}  
    </context>"""

    input_goes_here = """
    Input text:

    {text}
    """

    return general_instructions + "\n" + context_goes_here + "\n" + input_goes_here


# approved entities list can be used directly
schema_node_types = approved_entities

print("schema_node_types: ", schema_node_types)

# the keys from approved fact types dictionary can be used for relationship types
schema_relationship_types = [key.upper() for key in approved_fact_types.keys()]

print("schema_relationship_types: ", schema_relationship_types)

# rewrite the fact types into a list of tuples
schema_patterns = [
    [ fact['subject_label'], fact['predicate_label'].upper(), fact['object_label'] ]
    for fact in approved_fact_types.values()
]

print("schema_patterns:", schema_patterns)
# the complete entity schema
entity_schema = {
    "node_types": schema_node_types,
    "relationship_types": schema_relationship_types,
    "patterns": schema_patterns,
    "additional_node_types": False, # True would be less strict, allowing unknown node types
}