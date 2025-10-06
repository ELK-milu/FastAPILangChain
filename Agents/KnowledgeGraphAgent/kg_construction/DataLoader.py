# custom file data loader
import re
from pathlib import Path

from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo

class MarkdownDataLoader(DataLoader):
    def extract_title(self,markdown_text):
        # Define a regex pattern to match the first h1 header
        pattern = r'^# (.+)$'

        # Search for the first match in the markdown text
        match = re.search(pattern, markdown_text, re.MULTILINE)

        # Return the matched group if found
        return match.group(1) if match else "Untitled"

    async def run(self, filepath: Path, metadata = {}) -> PdfDocument:
        with open(filepath, "r") as f:
            markdown_text = f.read()
        doc_headline = self.extract_title(markdown_text)
        markdown_info = DocumentInfo(
            path=str(filepath),
            metadata={
                "title": doc_headline,
            }
        )
        return PdfDocument(text=markdown_text, document_info=markdown_info)


def file_context(file_path:str, num_lines=5) -> str:
    """Helper function to extract the first few lines of a file

    Args:
        file_path (str): Path to the file
        num_lines (int, optional): Number of lines to extract. Defaults to 5.

    Returns:
        str: First few lines of the file
    """
    with open(file_path, 'r') as f:
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    return "\n".join(lines)