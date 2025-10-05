import uuid

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import get_approved_files
from Agents.KnowledgeGraphAgent.UnstructuredDataAgent import ner_agent_instruction, ner_agent_initial_state
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.langgraph.Tools import get_approved_user_goal, sample_file
from utils.models import DeepSeek_V3

class UnstructuredDataAgentState(AgentState):
    proposed_entity_types : dict
    approved_entity_types : dict
    ner_agent_initial_state :dict


# tools to propose and approve entity types
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"

@tool(description="设置提议的实体类型列表，用于从非结构化文本中提取")
@needs_state
def set_proposed_entities(proposed_entity_types: list[str], state: UnstructuredDataAgentState = None) -> dict:
    """设置提议的实体类型列表，用于从非结构化文本中提取。

    Args:
        proposed_entity_types: 提议的实体类型名称列表
        state: Agent 状态对象

    Returns:
        dict: 包含操作元数据的字典。
              包含 'status' 键（'success' 或 'error'）。
              如果是 'success'，包含 'proposed_entity_types' 键以及设置的实体类型列表。
    """
    state[PROPOSED_ENTITIES] = proposed_entity_types
    return {
        "status": "success",
        PROPOSED_ENTITIES: proposed_entity_types
    }

@tool(description="获取提议的实体类型列表")
@needs_state
def get_proposed_entities(state: UnstructuredDataAgentState = None) -> dict:
    """获取提议的实体类型列表，用于从非结构化文本中提取。

    Args:
        state: Agent 状态对象

    Returns:
        dict: 包含 'proposed_entity_types' 键的字典，值为实体类型列表
    """
    return {
        "status": "success",
        PROPOSED_ENTITIES: state.get(PROPOSED_ENTITIES, [])
    }

@tool(description="批准提议的实体类型列表")
@needs_state
def approve_proposed_entities(state: UnstructuredDataAgentState = None) -> dict:
    """在用户批准后，将提议的实体类型记录为已批准的实体类型列表。

    仅在用户明确批准建议的实体类型时调用此工具。

    Args:
        state: Agent 状态对象

    Returns:
        dict: 包含操作元数据的字典。
              包含 'status' 键（'success' 或 'error'）。
              如果是 'success'，包含 'approved_entity_types' 键以及已批准的实体类型列表。
              如果是 'error'，包含 'error_message' 键。
    """
    if PROPOSED_ENTITIES not in state:
        return {
            "status": "error",
            "error_message": "没有可批准的提议实体类型。请先设置提议的实体类型，征求用户批准后再调用此工具。"
        }

    state[APPROVED_ENTITIES] = state.get(PROPOSED_ENTITIES)
    return {
        "status": "success",
        APPROVED_ENTITIES: state[APPROVED_ENTITIES]
    }

@tool(description="获取已批准的实体类型列表")
@needs_state
def get_approved_entities(state: UnstructuredDataAgentState = None) -> dict:
    """获取已批准的实体类型列表，用于从非结构化文本中提取。

    Args:
        state: Agent 状态对象

    Returns:
        dict: 包含 'approved_entity_types' 键的字典，值为已批准的实体类型列表
    """
    return {
        "status": "success",
        APPROVED_ENTITIES: state.get(APPROVED_ENTITIES, [])
    }

@tool(description="获取图模式中已批准的已知实体类型标签")
@needs_state
def get_well_known_types(state: UnstructuredDataAgentState = None) -> dict:
    """获取图模式中已批准的已知实体类型标签。

    从已批准的构建计划中提取所有节点类型的标签。

    Args:
        state: Agent 状态对象

    Returns:
        dict: 包含 'approved_labels' 键的字典，值为已批准标签的集合
    """
    construction_plan = state.get("approved_construction_plan", {})
    # approved labels are the keys for each construction plan entry where `construction_type` is "node"
    approved_labels = [entry["label"] for entry in construction_plan.values() if entry.get("construction_type") == "node"]
    return {
        "status": "success",
        "approved_labels": approved_labels
    }


tools = [get_approved_user_goal, get_approved_files, get_well_known_types,
         sample_file, set_proposed_entities, get_proposed_entities,
         approve_proposed_entities, get_approved_entities]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node_with_state(tools)
call_chat_node = create_chat_node(chat_model, ner_agent_instruction)
workflow = StateGraph(UnstructuredDataAgentState)
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


if __name__ == "__main__":
    # 使用 config["configurable"] 传递全局配置（推荐方案）
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }

    # inputs 只包含业务状态数据
    inputs = {
        "messages": [("user", "Add product reviews to the knowledge graph to trace product complaints back through the manufacturing process?")],
        "ner_agent_initial_state" : ner_agent_initial_state
    }

    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=graph,
        config=config,
        inputs=inputs,
        debug=False
    )

