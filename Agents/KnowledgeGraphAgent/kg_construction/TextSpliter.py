import re

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


# Define a custom text splitter. Chunking strategy could be yet-another-agent
class RegexTextSplitter(TextSplitter):
    """Split text using regex matched delimiters."""

    def __init__(self, re: str):
        self.re = re

    async def run(self, text: str) -> TextChunks:
        """Splits a piece of text into chunks.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of chunks.
        """
        texts = re.split(self.re, text)
        i = 0
        chunks = [TextChunk(text=str(text), index=i) for (i, text) in enumerate(texts)]
        return TextChunks(chunks=chunks)

