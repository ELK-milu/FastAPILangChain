from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# for example, creating a KG pipeline requires these arguments
if False:
    example = SimpleKGPipeline(
        llm=None,  # the LLM to use for Entity and Relation extraction
        driver=None,  # a neo4j driver to write results to graph
        embedder=None,  # an Embedder for chunks
        from_pdf=True,  # sortof True because you will use a custom loader
        pdf_loader=None,  # the custom loader for Markdown
        text_splitter=None,  # the splitter you defined above
        schema=None,  # that you just defined above
        prompt_template=None,  # the template used for entity extraction on each chunk
    )