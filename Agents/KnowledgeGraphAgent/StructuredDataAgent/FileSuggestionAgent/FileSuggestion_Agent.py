import uuid
from itertools import islice
from pathlib import Path

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from Agents.KnowledgeGraphAgent.StructuredDataAgent.FileSuggestionAgent import file_suggestion_agent_instruction
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node
from utils.langgraph.Tools import get_approved_user_goal, sample_file
from utils.models import DeepSeek_V3
from utils.neo4j import get_neo4j_import_dir

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
    import_dir = Path(get_neo4j_import_dir())

    file_names = [str(x.relative_to(import_dir).absolute())
                 for x in import_dir.rglob("*")
                 if x.is_file()]

    return {"status": "success", ALL_AVAILABLE_FILES: file_names}


# Tool: Set/Get suggested files
SUGGESTED_FILES = "suggested_files"

# 用于存储建议文件的全局状态（简化实现）
_file_suggestion_state = {}

@tool(description="设置建议用于数据导入的文件列表")
def set_suggested_files(suggest_files: list[str]) -> dict:
    f"""
    设置建议用于数据导入的文件列表。

    Args:
        suggest_files: 建议的文件路径列表

    Returns:
        dict: 一个包含内容元数据的字典。包含一个 'status' 键（'success' 或 'error'）。
            如果 'success'，则包含一个 {SUGGESTED_FILES} 键，值为文件名列表。
            如果 'error'，则包含一个 'error_message' 键。'error_message' 可能包含有关如何处理错误的说明。
    """
    _file_suggestion_state[SUGGESTED_FILES] = suggest_files
    return {"status": "success", SUGGESTED_FILES: suggest_files}


@tool(description="获取建议用于数据导入的文件列表")
def get_suggested_files() -> dict:
    f"""
    获取建议用于数据导入的文件列表。

    Returns:
        dict: 一个包含内容元数据的字典。包含一个 'status' 键（'success' 或 'error'）。
            如果 'success'，则包含一个 {SUGGESTED_FILES} 键，值为文件名列表。
            如果 'error'，则包含一个 'error_message' 键。
    """
    if SUGGESTED_FILES not in _file_suggestion_state:
        return {"status": "error", "error_message": "尚未设置建议文件。请先使用 set_suggested_files 工具。"}

    return {"status": "success", SUGGESTED_FILES: _file_suggestion_state[SUGGESTED_FILES]}


# Tool: Approve Suggested Files
APPROVED_FILES = "approved_files"


@tool(description="批准建议的文件以供进一步处理")
def approve_suggested_files() -> dict:
    f"""
    批准建议的文件以供进一步处理。将 {SUGGESTED_FILES} 转换为 {APPROVED_FILES}。

    Returns:
        dict: 一个包含内容元数据的字典。包含一个 'status' 键（'success' 或 'error'）。
            如果 'success'，则包含一个 {APPROVED_FILES} 键，值为批准的文件名列表。
            如果 'error'，则包含一个 'error_message' 键。'error_message' 可能包含有关如何处理错误的说明。
    """
    if SUGGESTED_FILES not in _file_suggestion_state:
        return {"status": "error", "error_message": "当前文件尚未设置。除了通知用户外不要采取任何行动。"}

    _file_suggestion_state[APPROVED_FILES] = _file_suggestion_state[SUGGESTED_FILES]
    return {"status": "success", APPROVED_FILES: _file_suggestion_state[APPROVED_FILES]}

@tool(description="获取已批准的文件列表")
def get_approved_files() -> list[str]:
    """
    获取已批准的文件列表。
    Returns:
        list[str]: 已批准的文件列表。
    """
    if APPROVED_FILES not in _file_suggestion_state:
        return []
    return _file_suggestion_state[APPROVED_FILES]



tools = [get_approved_user_goal, list_available_files, sample_file,
         set_suggested_files, get_suggested_files,
         approve_suggested_files
         ]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node(tools)

call_chat_node = create_chat_node(chat_model, file_suggestion_agent_instruction)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

# 添加条件边：agent 决定是否调用工具
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # 调用工具
        "end": END,           # 结束对话
    },
)

# 工具执行后返回 agent 继续处理
workflow.add_edge("tools", "agent")

checkpointer = InMemorySaver()

# 编译 workflow
graph = workflow.compile(checkpointer)
config = {"configurable": {"thread_id": uuid.uuid4()}}
inputs = {"messages": [("user", "我们可以使用哪些文件进行导入？")]}
result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
    graph=graph,
    config=config,
    inputs=inputs,
    debug=False
)
