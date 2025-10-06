import uuid

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import get_approved_files
from Agents.KnowledgeGraphAgent.UnstructuredDataAgent.EntityTypeProposalAgent.EntityTypeProposalAgent import \
    get_approved_entities
from Agents.KnowledgeGraphAgent.UnstructuredDataAgent.FactTypeExtractionSubAgent import fact_agent_instruction
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import needs_state, create_tool_node_with_state
from utils.langgraph.Tools import get_approved_user_goal, sample_file
from utils.models import DeepSeek_V3


class FactTypeExtractionSubAgentState(AgentState):
    proposed_fact_types : dict
    approved_fact_types : list[str]

PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"

@tool(description="添加一个可从文件中提取的提议事实类型")
@needs_state
def add_proposed_fact(approved_subject_label: str,
                      proposed_predicate_label: str,
                      approved_object_label: str,
                      state) -> dict:
    """
    添加一个可从文件中提取的提议事实类型。
    提议的事实类型是一个（主语，谓语，宾语）三元组，
    其中主语和宾语是已批准的实体类型，谓语是提议的关系标签。

    参数:
      approved_subject_label: 主语实体类型的已批准标签
      proposed_predicate_label: 谓语的关系标签
      approved_object_label: 宾语实体类型的已批准标签

    返回:
        dict: 包含操作元数据的字典。
                包含'status'键（值为'success'或'error'）。
                若状态为'success'，则包含'proposed_fact_added'键，其值为添加的事实类型详情
                若状态为'error'，则包含'error_message'键。
                'error_message'中可能包含有关如何处理错误的说明。
    """
    approved_entities = state.get("approved_fact_types", [])

    if approved_subject_label not in approved_entities:
        return {"status": "error", "error_message": f"Approved subject label {approved_subject_label} not found. Try again."}
    if approved_object_label not in approved_entities:
        return {"status": "error", "error_message": f"Approved object label {approved_object_label} not found. Try again."}

    current_predicates = state.get(PROPOSED_FACTS, {})
    current_predicates[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    state[PROPOSED_FACTS] = current_predicates
    return {"status": "success", "proposed_fact_added": current_predicates[proposed_predicate_label]}


@tool(description="获取可从文件中提取的提议事实类型")
@needs_state
def get_proposed_facts(state) -> dict:
    """获取可从文件中提取的提议事实类型"""
    return state.get(PROPOSED_FACTS, {})

@tool(description="在用户批准后，将提议的事实类型记录为已批准的事实类型,仅当用户明确批准了提议的事实类型时，才调用此工具。")
@needs_state
def approve_proposed_facts(state) -> dict:
    """在用户批准后，将提议的事实类型记录为已批准的事实类型
    仅当用户明确批准了提议的事实类型时，才调用此工具。
    """
    if PROPOSED_FACTS not in state:
        return {"status": "error", "error_message": "No proposed fact types to approve. Please set proposed facts first."}
    state[APPROVED_FACTS] = state.get(PROPOSED_FACTS, {})
    return {"status": "success", "approved_fact_types": state[APPROVED_FACTS]}


tools = [
    get_approved_user_goal, get_approved_files,
    get_approved_entities,
    sample_file,
    add_proposed_fact,
    get_proposed_facts,
    approve_proposed_facts
]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node_with_state(tools)
call_chat_node = create_chat_node(chat_model, fact_agent_instruction)
workflow = StateGraph(FactTypeExtractionSubAgentState)
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
graph = workflow.compile(checkpointer,debug=True)


if __name__ == "__main__":
    # 使用 config["configurable"] 传递全局配置（推荐方案）
    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }

    # inputs 只包含业务状态数据
    inputs = {
        "messages": [("user", "Propose fact types that can be found in the text.")],
    }

    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=graph,
        config=config,
        inputs=inputs,
        debug=False
    )

