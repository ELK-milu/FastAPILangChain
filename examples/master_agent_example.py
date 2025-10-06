"""
示例：主 Agent 调用子 Agent 的完整实现

展示如何将 FileSuggestionAgent 和 SchemaProposalAgent 封装为工具，
供一个协调型的 Master Agent 调用。
"""

import uuid
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
from utils.models import DeepSeek_V3


# ========== 1. 定义主 Agent 的状态 ==========
class MasterAgentState(AgentState):
    """主 Agent 的状态，用于协调多个子 Agent"""
    user_goal: str  # 用户目标
    approved_files: list  # 文件建议 Agent 返回的文件列表
    construction_plan: dict  # Schema 提案 Agent 返回的构建计划


# ========== 2. 封装子 Agent 为工具 ==========

@tool(description="调用文件建议 Agent，获取推荐用于知识图谱构建的文件列表")
@needs_state
def get_recommended_files(user_query: str, state: MasterAgentState = None) -> dict:
    """
    调用 FileSuggestionAgent 子 Agent，获取推荐的导入文件。

    Args:
        user_query: 用户对文件的需求描述
        state: 主 Agent 的状态对象

    Returns:
        dict: 包含 status 和 approved_files 的字典
    """
    # 导入子 Agent（延迟导入避免循环依赖）
    from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import (
        graph as file_suggestion_graph,
        get_approved_files
    )

    # 为子 Agent 创建独立的配置
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())  # 每次调用使用新的 thread ID
        }
    }

    # 构造子 Agent 的输入
    inputs = {
        "messages": [("user", user_query)]
    }

    try:
        print("\n[MasterAgent] 正在调用文件建议 Agent...")

        # 调用子 Agent（同步执行）
        result = file_suggestion_graph.invoke(inputs, config=config)

        # 方式1: 从返回的状态中提取结果
        # approved_files = result.get("approved_files", [])

        # 方式2: 调用子 Agent 提供的访问函数（如果使用了全局状态）
        approved_files = get_approved_files()

        # 更新主 Agent 的状态
        state["approved_files"] = approved_files

        print(f"[MasterAgent] 文件建议 Agent 返回 {len(approved_files)} 个文件")

        return {
            "status": "success",
            "approved_files": approved_files,
            "message": f"成功获取 {len(approved_files)} 个推荐文件"
        }

    except Exception as e:
        print(f"[MasterAgent] 调用文件建议 Agent 失败: {e}")
        return {
            "status": "error",
            "error_message": f"调用文件建议 Agent 失败: {str(e)}"
        }


@tool(description="调用 Schema 提案 Agent，为已批准的文件生成知识图谱构建方案")
@needs_state
def generate_construction_plan(state: MasterAgentState = None) -> dict:
    """
    调用 SchemaProposalAgent 子 Agent，生成知识图谱构建方案。

    此工具会从主 Agent 状态中读取 approved_files 和 user_goal，
    并将这些上下文传递给子 Agent。

    Args:
        state: 主 Agent 的状态对象

    Returns:
        dict: 包含 status 和 construction_plan 的字典
    """
    # 导入子 Agent
    from Agents.KnowledgeGraphAgent.StructuredDataAgent.SchemaProposalAgent.SchemaProposalAgent import (
        graph as schema_proposal_graph
    )

    # 从主状态获取上下文信息
    approved_files = state.get("approved_files", [])
    user_goal = state.get("user_goal", "构建知识图谱")

    if not approved_files:
        return {
            "status": "error",
            "error_message": "没有已批准的文件。请先调用 get_recommended_files 工具。"
        }

    # 为子 Agent 创建独立的配置
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    # 构造子 Agent 的输入，包含上下文信息
    inputs = {
        "messages": [
            ("system", f"用户目标: {user_goal}"),
            ("system", f"已批准的文件: {', '.join(approved_files)}"),
            ("user", "请为这些文件生成知识图谱的节点和关系构建方案。")
        ]
    }

    try:
        print("\n[MasterAgent] 正在调用 Schema 提案 Agent...")

        # 调用子 Agent
        result = schema_proposal_graph.invoke(inputs, config=config)

        # 提取构建计划
        construction_plan = result.get("approved_construction_plan", {})

        # 更新主 Agent 的状态
        state["construction_plan"] = construction_plan

        print(f"[MasterAgent] Schema 提案 Agent 返回了构建方案")

        return {
            "status": "success",
            "construction_plan": construction_plan,
            "message": f"成功生成包含 {len(construction_plan)} 个构建规则的方案"
        }

    except Exception as e:
        print(f"[MasterAgent] 调用 Schema 提案 Agent 失败: {e}")
        return {
            "status": "error",
            "error_message": f"调用 Schema 提案 Agent 失败: {str(e)}"
        }


@tool(description="获取当前用户的目标")
@needs_state
def get_user_goal_from_state(state: MasterAgentState = None) -> dict:
    """
    从主 Agent 状态中获取用户目标。

    Args:
        state: 主 Agent 的状态对象

    Returns:
        dict: 包含 user_goal 的字典
    """
    user_goal = state.get("user_goal", "未设置")
    return {
        "status": "success",
        "user_goal": user_goal
    }


# ========== 3. 构建主 Agent ==========

# 主 Agent 可用的工具列表
tools = [
    get_recommended_files,
    generate_construction_plan,
    get_user_goal_from_state
]

# 绑定工具到模型
chat_model = DeepSeek_V3.bind_tools(tools)

# 创建工具节点（支持状态传递）
tool_node = create_tool_node_with_state(tools)

# 主 Agent 的系统指令
master_agent_instruction = """你是知识图谱构建的协调 Agent（Master Agent）。

你的职责是：
1. 理解用户的知识图谱构建需求
2. 调用 get_recommended_files 工具来获取推荐的导入文件
3. 调用 generate_construction_plan 工具为这些文件生成构建方案
4. 向用户报告完整的构建计划

工作流程：
- 首先，调用 get_user_goal_from_state 了解用户目标
- 然后，调用 get_recommended_files 获取推荐文件
- 等待用户确认文件列表
- 最后，调用 generate_construction_plan 生成详细的构建方案

注意：
- 你不直接处理文件或生成 Schema，而是协调两个专业的子 Agent
- 子 Agent 的调用结果会自动保存到你的状态中
- 始终向用户清晰报告每个步骤的进展
"""

# 创建聊天节点
call_chat_node = create_chat_node(chat_model, master_agent_instruction)

# 构建状态图
workflow = StateGraph(MasterAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

# 添加条件边：agent 决定是否调用工具
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # 继续调用工具
        "end": END,           # 结束对话
    },
)

# 工具执行后返回 agent 继续处理
workflow.add_edge("tools", "agent")

# 使用 checkpoint 支持状态持久化
checkpointer = InMemorySaver()

# 编译 workflow
master_graph = workflow.compile(checkpointer)


# ========== 4. 运行主 Agent ==========

if __name__ == "__main__":
    print("=" * 80)
    print("主 Agent 调用子 Agent 示例")
    print("=" * 80)

    # 配置（包含 thread_id 用于状态管理）
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    # 初始输入
    inputs = {
        "messages": [
            ("user", "我想构建一个电影推荐的知识图谱，帮我完成整个流程")
        ],
        "user_goal": "构建电影推荐知识图谱"
    }

    print(f"\n用户输入: {inputs['messages'][0][1]}")
    print(f"用户目标: {inputs['user_goal']}\n")

    try:
        # 使用带审批的工作流运行器
        # auto_approve=True 表示自动批准所有审批（测试用）
        result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
            graph=master_graph,
            config=config,
            inputs=inputs,
            auto_approve=False,  # 设为 False 启用人工审批
            debug=True           # 显示详细调试信息
        )

        print("\n" + "=" * 80)
        print("执行完成！")
        print("=" * 80)

        # 显示最终结果
        print(f"\n批准的文件数量: {len(result.get('approved_files', []))}")
        print(f"批准的文件: {result.get('approved_files', [])}")

        construction_plan = result.get('construction_plan', {})
        print(f"\n构建方案包含 {len(construction_plan)} 个规则:")
        for key, rule in construction_plan.items():
            print(f"  - {key}: {rule.get('construction_type', 'unknown')}")

        print(f"\nAgent 消息数量: {len(agent_msgs)}")
        print(f"工具调用数量: {len(tool_msgs)}")

    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n执行失败: {e}")
        import traceback
        traceback.print_exc()
