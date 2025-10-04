# Tool: List Import Files
import os
from itertools import islice
from pathlib import Path

from langchain_core.tools import tool

from utils.neo4j import get_neo4j_import_dir

# this constant will be used as the key for storing the file list in the tool context state
ALL_AVAILABLE_FILES = "all_available_files"
@tool(description="列出可用于知识图谱构建的文件。所有文件路径均相对于导入目录。")
def list_available_files() -> dict:
    f"""
    列出可用于知识图谱构建的文件。所有文件路径均相对于导入目录。
    Returns:
        dict: 一个包含内容元数据的字典。包含一个 'status' 键（'success' 或 'error'）。
            如果 'success'，则包含一个 {ALL_AVAILABLE_FILES} 键，值为文件名列表。
            如果 'error'，则包含一个 'error_message' 键。'error_message' 可能包含有关如何处理错误的说明。  
    """
    # get the import dir using the helper function
    import_dir = Path(get_neo4j_import_dir())

    # get a list of relative file names, so files must be rooted at the import dir
    file_names = [str(x.relative_to(import_dir))
                 for x in import_dir.rglob("*")
                 if x.is_file()]

    return {"status": "success", ALL_AVAILABLE_FILES: file_names}


@tool(description="这是一个简单的文件读取工具，仅适用于导入目录中的文件")
def sample_file(file_path: str) -> dict:

    """
    通过读取文件内容作为文本来对文件进行采样。将任何文件视为文本，并最多读取100行。
        Args:
            file_path: 要采样的文件，相对于导入目录的路径返回值
        Returns:
            dict: 一个包含内容元数据的字典，以及文件的采样。包含一个 'status' 键（'success' 或 'error'）。
                如果 'success'，包含一个 'content' 键，存储文本文件内容。
                如果 'error'，包含一个 'error_message' 键。
                'error_message' 可能包含有关如何处理错误的说明。
    """
    # Trust, but verify. The agent may invent absolute file paths.
    if Path(file_path).is_absolute():
        return {"status": "error", "error_message": "File path must be relative to import directory"}
    import_dir = Path(get_neo4j_import_dir())
    # create the full path by extending from the import_dir
    full_path_to_file = import_dir / file_path

    # of course, _that_ may not exist
    if not full_path_to_file.exists():
        return {"status": "error", "error_message": f"File {file_path} does not exist"}
    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return {"status": "success", "content": content}

    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# Tool: Set/Get suggested files
SUGGESTED_FILES = "suggested_files"

def set_suggested_files(suggest_files:List[str], tool_context:ToolContext) -> Dict[str, Any]:
    """Set the suggested files to be used for data import.

    Args:
        suggest_files (List[str]): List of file paths to suggest

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)

# Helps encourage the LLM to first set the suggested files.
# This is an important strategy for maintaining consistency through defined values.
def get_suggested_files(tool_context:ToolContext) -> Dict[str, Any]:
    """Get the files to be used for data import.

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
    """
    return tool_success(SUGGESTED_FILES, tool_context.state[SUGGESTED_FILES])


# Tool: Approve Suggested Files
# Just like the previous lesson, you'll define a tool which
# accepts no arguments and can sanity check before approving.
APPROVED_FILES = "approved_files"


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Approves the {SUGGESTED_FILES} in state for further processing as {APPROVED_FILES}.

    If {SUGGESTED_FILES} is not in state, return an error.
    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])