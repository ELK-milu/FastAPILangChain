# 定义工具调用节点
import inspect
import json
from typing import Literal, Dict, Any, Callable, Optional, List
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import interrupt, Command

# 用于存储需要 state 的工具函数
_TOOLS_NEED_STATE = set()


def needs_state(tool_func):
    """
    装饰器，标记工具函数需要访问 state

    使用方法：
    @tool(description="工具描述")
    @needs_state
    def my_tool(param: str, state: MyState) -> dict:
        return {"status": "success"}

    或者在定义后调用：
    @tool(description="工具描述")
    def my_tool(param: str, state: MyState) -> dict:
        return {"status": "success"}

    my_tool = needs_state(my_tool)
    """
    # 记录原始函数名（在被 @tool 包装前的函数）
    func_to_mark = getattr(tool_func, 'func', tool_func)
    _TOOLS_NEED_STATE.add(id(func_to_mark))
    return tool_func


def create_tool_node(tools):
    """
    创建一个通用的工具调用节点
    :param tools: 工具列表
    """
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state):
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

    使用方法：
    1. 使用 @needs_state 装饰器标记工具
    2. 在工具函数签名中声明 state 参数
    3. 工具函数可以读取和修改 state

    示例：
    @tool(description="需要访问 state 的工具")
    @needs_state
    def my_tool(user_input: str, state: MyState) -> dict:
        old_value = state.get("key")
        state["key"] = "new_value"
        return {"status": "success"}

    或者在定义后标记：
    @tool(description="需要访问 state 的工具")
    def my_tool(user_input: str, state: MyState) -> dict:
        return {"status": "success"}

    my_tool = needs_state(my_tool)

    :param tools: 工具列表
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            tool_args = tool_call["args"]

            # 检查工具是否需要 state（通过全局集合检查）
            original_func = tool.func
            tool_needs_state = id(original_func) in _TOOLS_NEED_STATE

            if tool_needs_state:
                # 直接调用底层函数并传入 state
                # 检查原始函数签名
                func_signature = inspect.signature(original_func)

                if "state" in func_signature.parameters:
                    # 从 tool_args 中移除 state（如果 LLM 错误地传递了它）
                    clean_args = {k: v for k, v in tool_args.items() if k != "state"}
                    # 如果原始函数接受 state 参数，直接调用并注入真实的 state
                    tool_result = original_func(**clean_args, state=state)
                else:
                    # 如果不接受 state 参数，抛出错误提示
                    raise ValueError(
                        f"工具 {tool.name} 被标记为 needs_state，但函数签名中没有 state 参数。"
                        f"请在函数定义中添加 state 参数：def {original_func.__name__}(..., state: AgentState)"
                    )
            else:
                # 普通工具，使用标准 invoke
                tool_result = tool.invoke(tool_args)

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        # 构建返回值：包含 messages 和所有被修改的自定义字段
        result = {"messages": outputs}

        # LangGraph 管理字段列表（这些字段由 LangGraph 自动管理，节点不应返回）
        MANAGED_FIELDS = {"remaining_steps", "is_last_step"}

        # 将 state 中除了 messages 和管理字段之外的所有字段都返回
        # 这样可以保留工具函数对 state 的修改
        for key in state.keys():
            if key != "messages" and key not in MANAGED_FIELDS:
                result[key] = state[key]

        print(f"[DEBUG ToolNode] 返回值 keys: {list(result.keys())}")
        return result

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