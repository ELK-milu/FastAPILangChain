# 定义工具调用节点
import inspect
import json
from typing import Literal, Dict, Any, Callable, Optional, List
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import interrupt, Command


def create_tool_node(tools):
    """
    创建一个通用的工具调用节点
    :param tools: 工具列表
    """
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    return tool_node


def create_tool_node_with_state(tools):
    """
    创建一个可以将 state 传递给工具的工具调用节点
    :param tools: 工具列表
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            tool_args = tool_call["args"]

            # 检查工具是否接受 state 参数
            tool_signature = inspect.signature(tool.func)
            if "state" in tool_signature.parameters:
                # 如果工具签名包含 state 参数,则传递
                tool_args = {**tool_args, "state": state}

            tool_result = tool.invoke(tool_args)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    return tool_node



def create_tool_node_with_approval(
        tools: List,
        approved_node_name: str,
        rejected_node_name: str,
        approval_strategy: Literal["before", "after"] = "before",
        should_approve_fn: Optional[Callable] = None,
        interrupt_dict: Optional[Dict[str, Any]] = None,
        custom_interrupt_logic: Optional[Callable] = None
):
    """
    创建一个带人工审批的工具调用节点

    Args:
        tools: 工具列表
        approved_node_name: 批准后转向的节点名称
        rejected_node_name: 拒绝后转向的节点名称
        approval_strategy: 审批策略，"before"表示执行前审批，"after"表示执行后审批
        should_approve_fn: 自定义函数，判断是否需要审批（返回True需要审批）
        interrupt_dict: 中断时传递的固定信息字典
        custom_interrupt_logic: 自定义中断逻辑函数，接收state和tool_calls作为参数

    Returns:
        返回一个可用于LangGraph的节点函数
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node_with_approval(state: AgentState) -> Command[Literal[approved_node_name, rejected_node_name]]:
        # 获取最后一条消息中的工具调用
        last_message = state["messages"][-1]
        print(f"正在处理工具调用")
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            # 没有工具调用，直接继续
            print(f"没有工具调用，直接继续")
            return Command(goto=approved_node_name, update={"messages": []})

        tool_calls = last_message.tool_calls

        # 判断是否需要审批
        print(f"判断是否需要审批")
        needs_approval = True
        if should_approve_fn:
            needs_approval = should_approve_fn(state, tool_calls)

        # ==================== 执行前审批 ====================
        if approval_strategy == "before" and needs_approval:
            print(f"执行前审批")
            # 构建中断信息
            interrupt_info = {
                "question": "是否批准执行以下工具调用？",
                "tool_calls": [
                    {
                        "tool_name": tc["name"],
                        "arguments": tc["args"]
                    } for tc in tool_calls
                ],
                "state_summary": f"当前消息数: {len(state['messages'])}"
            }

            # 添加固定的中断信息
            if interrupt_dict:
                interrupt_info.update(interrupt_dict)

            # 如果有自定义逻辑，使用它来构建中断信息
            if custom_interrupt_logic:
                custom_info = custom_interrupt_logic(state, tool_calls)
                if isinstance(custom_info, dict):
                    interrupt_info.update(custom_info)

            # 触发中断等待人工审批
            is_approved = interrupt(interrupt_info)

            if not is_approved:
                print(f"❌ 工具调用被拒绝，转向节点: {rejected_node_name}")
                return Command(
                    goto=rejected_node_name,
                    update={"messages": [AIMessage(content="工具调用已被拒绝")]}
                )

            print(f"✅ 工具调用已批准，开始执行")

        # ==================== 执行工具 ====================
        outputs = []
        try:
            print(f"执行工具")
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                if tool_name not in tools_by_name:
                    outputs.append(
                        ToolMessage(
                            content=json.dumps({"error": f"工具 {tool_name} 未找到"}),
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

                # 执行工具
                tool_result = tools_by_name[tool_name].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                    )
                )
        except Exception as e:
            outputs.append(
                ToolMessage(
                    content=json.dumps({"error": str(e)}),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        # ==================== 执行后审批 ====================
        if approval_strategy == "after" and needs_approval:
            print(f"执行后审批")
            # 构建中断信息
            interrupt_info = {
                "question": "是否批准以下工具执行结果？",
                "tool_results": [
                    {
                        "tool_name": msg.name,
                        "result": msg.content
                    } for msg in outputs
                ],
                "state_summary": f"当前消息数: {len(state['messages'])}"
            }

            # 添加固定的中断信息
            if interrupt_dict:
                interrupt_info.update(interrupt_dict)

            # 如果有自定义逻辑
            if custom_interrupt_logic:
                custom_info = custom_interrupt_logic(state, tool_calls, outputs)
                if isinstance(custom_info, dict):
                    interrupt_info.update(custom_info)

            # 触发中断等待人工审批
            is_approved = interrupt(interrupt_info)

            if not is_approved:
                print(f"❌ 工具执行结果被拒绝，转向节点: {rejected_node_name}")
                return Command(
                    goto=rejected_node_name,
                    update={"messages": [AIMessage(content="工具执行结果已被拒绝")]}
                )

            print(f"✅ 工具执行结果已批准")

        # 返回结果并转向批准节点
        return Command(goto=approved_node_name, update={"messages": outputs})

    return tool_node_with_approval


# ==================== 使用示例 ====================

# 示例1: 基本使用 - 执行前审批
def example_basic():
    """最简单的使用方式"""
    from langchain_core.tools import tool

    @tool
    def dangerous_operation(data: str):
        """执行危险操作"""
        return {"status": "success", "data": data}

    tools = [dangerous_operation]

    tool_node = create_tool_node_with_approval(
        tools=tools,
        approved_node_name="agent",  # 批准后返回agent
        rejected_node_name="rejected_node",  # 拒绝后转向rejected节点
        approval_strategy="before"  # 执行前审批
    )

    return tool_node


# 示例2: 条件审批 - 只对特定工具审批
def example_conditional_approval():
    """只对特定工具进行审批"""
    from langchain_core.tools import tool

    @tool
    def safe_operation(data: str):
        """安全操作"""
        return {"status": "success"}

    @tool
    def dangerous_operation(data: str):
        """危险操作"""
        return {"status": "success"}

    tools = [safe_operation, dangerous_operation]

    # 自定义审批条件：只对dangerous_operation进行审批
    def should_approve(state, tool_calls):
        for tc in tool_calls:
            if tc["name"] == "dangerous_operation":
                return True
        return False

    tool_node = create_tool_node_with_approval(
        tools=tools,
        approved_node_name="agent",
        rejected_node_name="rejected_node",
        should_approve_fn=should_approve,
        approval_strategy="before"
    )

    return tool_node


# 示例3: 自定义中断信息
def example_custom_interrupt():
    """自定义中断时显示的信息"""
    from langchain_core.tools import tool

    @tool
    def set_perceived_user_goal(kind_of_graph: str, graph_description: str):
        """设定感知用户目标"""
        return {"kind_of_graph": kind_of_graph, "graph_description": graph_description}

    tools = [set_perceived_user_goal]

    # 自定义中断逻辑
    def custom_logic(state, tool_calls):
        tool_call = tool_calls[0]
        return {
            "graph_type": tool_call["args"].get("kind_of_graph", "未知"),
            "description": tool_call["args"].get("graph_description", "未知"),
            "提示": "请仔细检查图表类型和描述是否准确"
        }

    tool_node = create_tool_node_with_approval(
        tools=tools,
        approved_node_name="approval_node",
        rejected_node_name="rejected_node",
        interrupt_dict={"priority": "high", "category": "graph_definition"},
        custom_interrupt_logic=custom_logic,
        approval_strategy="before"
    )

    return tool_node


# 示例4: 执行后审批
def example_after_approval():
    """在工具执行后进行审批"""
    from langchain_core.tools import tool

    @tool
    def query_database(query: str):
        """查询数据库"""
        return {"result": "查询结果...", "row_count": 100}

    tools = [query_database]

    # 自定义逻辑：检查结果
    def custom_logic(state, tool_calls, outputs=None):
        if outputs:
            result_data = json.loads(outputs[0].content)
            return {
                "结果预览": result_data,
                "提示": "请确认查询结果是否符合预期"
            }
        return {}

    tool_node = create_tool_node_with_approval(
        tools=tools,
        approved_node_name="agent",
        rejected_node_name="rejected_node",
        custom_interrupt_logic=custom_logic,
        approval_strategy="after"  # 执行后审批
    )

    return tool_node