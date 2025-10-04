# Tool: List Import Files
import os
from pathlib import Path

from langchain_core.tools import tool



# this constant will be used as the key for storing the file list in the tool context state
ALL_AVAILABLE_FILES = "all_available_files"
@tool
def list_available_files(tool_context:ToolContext) -> dict:
    f"""
    列出可用于知识图谱构建的文件。所有文件路径均相对于导入目录。
    Returns:
        dict: 一个包含内容元数据的字典。包含一个 'status' 键（'success' 或 'error'）。
            如果 'success'，则包含一个 {ALL_AVAILABLE_FILES} 键，值为文件名列表。
            如果 'error'，则包含一个 'error_message' 键。'error_message' 可能包含有关如何处理错误的说明。  
    """
    # get the import dir using the helper function
    import_dir = get_neo4j_import_dir()

    # get a list of relative file names, so files must be rooted at the import dir
    file_names = [str(x.relative_to(import_dir))
                 for x in import_dir.rglob("*")
                 if x.is_file()]

    # save the list to state so we can inspect it later
    tool_context.state[ALL_AVAILABLE_FILES] = file_names

    return tool_success(ALL_AVAILABLE_FILES, file_names)