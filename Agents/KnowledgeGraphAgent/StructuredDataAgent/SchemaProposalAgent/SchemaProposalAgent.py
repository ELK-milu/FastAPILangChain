import uuid
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import get_approved_files
from Agents.KnowledgeGraphAgent.StructuredDataAgent.SchemaProposalAgent import proposal_agent_instruction
from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.langgraph.ToolNode import needs_state, create_tool_node_with_state
from utils.langgraph.Tools import get_approved_user_goal, sample_file, search_file
from utils.models import DeepSeek_V3

#  Tool: Propose Node Construction

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
NODE_CONSTRUCTION = "node_construction"

class SchemaProposalAgentState(AgentState):
    # 自定义字段使用 keep_latest Reducer 保持值的传递
    proposed_construction_plan : dict



@tool(description="为已批准的文件提议节点构建方案以支持用户目标")
@needs_state
def propose_node_construction(approved_file: str, proposed_label: str, unique_column_name: str, proposed_properties: list[str],state: SchemaProposalAgentState=None) -> dict:
    """
    为已批准的文件提议节点构建方案以支持用户目标。

    构建方案将被添加到提议的构建计划字典中，使用 proposed_label 作为键。

    构建条目将是一个包含以下键的字典：
    - construction_type: "node"
    - source_file: 要提议节点构建的已批准文件
    - label: 节点的提议标签
    - unique_column_name: 将用于唯一标识构建节点的列名
    - properties: 节点的属性名称列表，源自已批准文件中的列名

    Args:
        approved_file: 要提议节点构建的已批准文件
        proposed_label: 构建节点的提议标签（用作构建计划中的键）
        unique_column_name: 将用于唯一标识构建节点的列名
        proposed_properties: 应导入为节点属性的列名列表

    Return:
        dict: 包含内容元数据的字典。
              包含 'status' 键（'success' 或 'error'）。
              如果是 'success'，包含 'node_construction' 键以及节点的构建计划。
              如果是 'error'，包含 'error_message' 键。
              'error_message' 可能包含关于如何处理错误的说明。
    """
    # 快速健全性检查 -- 已批准的文件是否有唯一列？
    search_results = search_file.invoke(
        {
            "file_path": approved_file,
            "query": unique_column_name
        })
    if search_results["status"] == "error":
        return search_results # return the error
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return { "status": "error", "error_message": f"{approved_file} 不包含列 {unique_column_name}。请检查文件内容并重试。"}
    # get the current construction plan, or an empty one if none exists
    construction_plan = state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }
    construction_plan[proposed_label] = node_construction_rule
    state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return {
        "status": "success",
        NODE_CONSTRUCTION: node_construction_rule
    }


RELATIONSHIP_CONSTRUCTION = "relationship_construction"
@tool(description="为已批准的文件提议关系构建方案以支持用户目标")
@needs_state
def propose_relationship_construction(approved_file: str, proposed_relationship_type: str,
    from_node_label: str,from_node_column: str, to_node_label:str, to_node_column: str,
    proposed_properties: list[str],
    state:SchemaProposalAgentState) -> dict:
    """
    为已批准文件提出一种关系构建方案，以支持用户目标。
    该构建方案将被添加到提议的构建计划字典中，使用 proposed_relationship_type 作为键。

    Args:
        approved_file: 需要为其提出节点构建方案的已批准文件
        proposed_relationship_type: 为构建的关系提议的标签名称
        from_node_label: 源节点的标签
        from_node_column: 已批准文件中用于唯一标识源节点的列名
        to_node_label: 目标节点的标签
        to_node_column: 已批准文件中用于唯一标识目标节点的列名
        unique_column_name: 用于唯一标识目标节点的列名

    Return:
        dict: 包含内容元数据的字典。
                包含'status'键（值为'success'或'error'）。
                若状态为'success'，则包含"relationship_construction"键，其中存储关系构建计划
                若状态为'error'，则包含'error_message'键。
                'error_message'中可能包含有关如何处理错误的说明。
    """
    # quick sanity check -- does the approved file have the from_node_column?
    search_results = search_file(approved_file, from_node_column)
    if search_results["status"] == "error":
        return search_results  # return the error if there is one
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return {"status": "error", "error_message": f"{approved_file} does not have the from node column {from_node_column}. Check the content of the file and reconsider the relationship."}

    # quick sanity check -- does the approved file have the to_node_column?
    search_results = search_file(approved_file, to_node_column)
    if search_results["status"] == "error" or search_results["search_results"]["metadata"]["lines_found"] == 0:
        return {"status": "error", "error_message": f"{approved_file} does not have the to node column {to_node_column}. Check the content of the file and reconsider the relationship."}

    construction_plan = state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }
    construction_plan[proposed_relationship_type] = relationship_construction_rule
    state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return {
        "status": "success",
        RELATIONSHIP_CONSTRUCTION: relationship_construction_rule
    }

@tool(description="根据关系类型从提议的构建计划中删除关系构建方案。")
@needs_state
def remove_node_construction(node_label: str, state:SchemaProposalAgentState) -> dict:
    """根据标签从提议的构建计划中移除节点构建方案。

    Args:
        node_label: 要移除的节点构建方案对应的标签
        tool_context: 工具上下文信息

    Return:
        dict: 包含操作元数据的字典。
                包含'status'键（值为'success'或'error'）。
                若状态为'success'，则包含'node_construction_removed'键，其值为被移除的节点构建方案的标签
                若状态为'error'，则包含'error_message'键。
                'error_message'中可能包含有关如何处理错误的说明。
    """
    construction_plan = state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if node_label not in construction_plan:
        return {"status": "error", "error_message": f"{node_label} not found in proposed construction plan."}

    del construction_plan[node_label]

    state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return {"status": "success", "node_construction_removed": node_label}

@tool(description="根据类型从提议的构建计划中移除关系构建方案")
@needs_state
def remove_relationship_construction(relationship_type: str, state:SchemaProposalAgentState) -> dict:
    """根据类型从提议的构建计划中移除关系构建方案。
    参数:
        relationship_type: 要移除的关系构建方案的类型
        tool_context: 工具上下文信息

    返回:
        dict: 包含操作元数据的字典。
                包含'status'键（值为'success'或'error'）。
                若状态为'success'，则包含'relationship_construction_removed'键，其值为被移除的关系构建方案的类型
                若状态为'error'，则包含'error_message'键。
                'error_message'中可能包含有关如何处理错误的说明。
            """
    construction_plan = state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    if relationship_type not in construction_plan:
        return {"status": "error", "error_message": f"{relationship_type} not found in proposed construction plan."}

    construction_plan.pop(relationship_type)

    state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return {"status": "success", "relationship_construction_removed": relationship_type}

@tool(description="获取提议的构建计划")
@needs_state
def get_proposed_construction_plan(state:SchemaProposalAgentState) -> dict:
    """获取提议的构建计划，即构建规则字典
    返回:
        dict: 包含proposed_construction_plan的字典
    """
    return state.get(PROPOSED_CONSTRUCTION_PLAN, {})


APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"
@tool(description="批准提议的构建计划")
@needs_state
def approve_proposed_construction_plan(state:SchemaProposalAgentState) -> dict:
    """批准提议的构建计划

    返回:
        dict: 返回一个APPROVED_CONSTRUCTION_PLAN的字典
    """
    if not PROPOSED_CONSTRUCTION_PLAN in state:
        return {"status": "error", "error_message": "No proposed construction plan found."}

    state[APPROVED_CONSTRUCTION_PLAN] = state.get(PROPOSED_CONSTRUCTION_PLAN)
    return {"status": "success", APPROVED_CONSTRUCTION_PLAN: state[APPROVED_CONSTRUCTION_PLAN]}



tools = [propose_node_construction,remove_node_construction,
         remove_relationship_construction,propose_relationship_construction,
         get_proposed_construction_plan,approve_proposed_construction_plan,
         get_approved_user_goal,get_approved_files,
         sample_file,search_file]

chat_model = DeepSeek_V3.bind_tools(
    tools
)
tool_node = create_tool_node_with_state(tools)


call_chat_node = create_chat_node(chat_model, proposal_agent_instruction)

workflow = StateGraph(SchemaProposalAgentState)
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
        "messages": [("user", "How can these files be imported?")],
    }

    print("\n=== 使用 config['configurable'] 传递全局配置 ===")
    print(f"全局配置: {config['configurable']}")
    print(f"业务输入: {inputs}\n")

    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=graph,
        config=config,
        inputs=inputs,
        debug=False
    )
