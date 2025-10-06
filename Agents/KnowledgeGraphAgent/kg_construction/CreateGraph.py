import os

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from Agents.KnowledgeGraphAgent import approved_files, graphdb
from Agents.KnowledgeGraphAgent.kg_construction import contextualize_er_extraction_prompt, llm_for_neo4j, neo4j_driver, \
    embedder, entity_schema
from Agents.KnowledgeGraphAgent.kg_construction.DataLoader import file_context, MarkdownDataLoader
from Agents.KnowledgeGraphAgent.kg_construction.TextSpliter import RegexTextSplitter
from utils.neo4j import get_neo4j_import_dir


def make_kg_builder(file_path:str) -> SimpleKGPipeline:
    """Builds a KG builder for a given file, which is used to contextualize the chunking and entity extraction."""
    context = file_context(file_path)
    contextualized_prompt = contextualize_er_extraction_prompt(context)

    return SimpleKGPipeline(
        llm=llm_for_neo4j, # the LLM to use for Entity and Relation extraction
        driver=neo4j_driver,  # a neo4j driver to write results to graph
        embedder=embedder,  # an Embedder for chunks
        from_pdf=True,   # sortof True because you will use a custom loader
        pdf_loader=MarkdownDataLoader(), # the custom loader for Markdown
        text_splitter=RegexTextSplitter("---"), # the splitter you defined above
        schema=entity_schema, # that you just defined above
        prompt_template=contextualized_prompt,
    )


neo4j_import_dir = get_neo4j_import_dir() or "."

async def main():
    for file_name in approved_files:
        file_path = os.path.join(neo4j_import_dir, file_name)
        print(f"Processing file: {file_name}")
        kg_builder = make_kg_builder(file_path)
        results = await kg_builder.run_async(file_path=str(file_path))
        print("\tResults:", results.result)
    print("All files processed.")
