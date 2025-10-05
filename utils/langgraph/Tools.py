from itertools import islice
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt

from utils.neo4j import get_neo4j_import_dir

SEARCH_RESULTS = "search_results"
@tool(description="为已批准的文件提议关系构建方案以支持用户目标")
def search_file(file_path: str, query: str) -> dict:
    """
    在任意文本文件（markdown、csv、txt）中搜索包含指定查询字符串的行。
    类似 grep 的简单功能，适用于任何文本文件。
    搜索始终不区分大小写。

    参数:
      file_path: 文件路径，相对于 Neo4j 导入目录。
      query: 要搜索的字符串。

    返回:
        dict: 包含 'status'（'success' 或 'error'）的字典。
              如果是 'success'，包含 'search_results'，其中包含 'matching_lines'
              （包含 'line_number' 和 'content' 键的字典列表）
              以及搜索的基本元数据。
              如果是 'error'，包含 'error_message'。
    """
    import_dir = Path(get_neo4j_import_dir())
    p = import_dir / file_path

    if not p.exists():
        return {"status": "error", "error_message": f"File not found: {file_path}"}
    if not p.is_file():
        return {"status": "error", "error_message": f"{file_path} is not a file"}

    # Handle empty query - return no results
    if not query:
        return {
            "status": "success",
            SEARCH_RESULTS: {
                "metadata": {
                    "path": file_path,
                    "query": query,
                    "lines_found": 0
                },
                "matching_lines": []
            }
        }

    matching_lines = []
    search_query = query.lower()

    try:
        with open(p, 'r', encoding='utf-8') as file:
            # Process the file line by line
            for i, line in enumerate(file, 1):
                line_to_check = line.lower()
                if search_query in line_to_check:
                    matching_lines.append({
                        "line_number": i,
                        "content": line.strip()  # Remove trailing newlines
                    })

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

    # Prepare basic metadata
    metadata = {
        "path": file_path,
        "query": query,
        "lines_found": len(matching_lines)
    }

    result_data = {
        "metadata": metadata,
        "matching_lines": matching_lines
    }
    return {
        "status": "success",
        SEARCH_RESULTS: result_data
    }


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

@tool(description="请求人工审批工具，用于在执行重要操作前获得人类确认。")
def get_approved_user_goal(
    operation_description: str,
):
    """
    请求人工审批，在执行重要操作前获得人类确认。

    Args:
        operation_description: 需要审批的操作描述，例如 "设定用户目标为BOM图表"
    Returns:
        dict: 包含审批结果的字典，格式为 {"approved": bool, "message": str}
    """
    # 构建详细的中断信息
    interrupt_info = {
        "question": f"是否批准以下操作？",
        "operation": operation_description,
    }

    # 触发人工审批
    is_approved = interrupt(interrupt_info)

    # 返回审批结果
    if is_approved:
        result = {
            "approved": True,
            "message": f"✅ 操作已批准：{operation_description}",
        }
        print(f"✅ 人工审批通过: {operation_description}")
    else:
        result = {
            "approved": False,
            "message": f"❌ 操作被拒绝：{operation_description}",
        }
        print(f"❌ 人工审批被拒绝: {operation_description}")

    return result