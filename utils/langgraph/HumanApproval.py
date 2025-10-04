from typing import Literal, TypedDict, Any, Dict, Callable, Optional
from langgraph.types import interrupt, Command


def HumanApproval(
        state_type: type,
        approved_node_name: str,
        rejected_node_name: str,
        interrupt_dict: Optional[Dict[str, Any]] = None,
        custom_interrupt_logic: Optional[Callable] = None
):
    """
    封装人类审批节点
    Args:
        state_type: 状态类型
        interrupt_dict: 中断时传递的固定信息字典
        approved_node_name: 批准后转向的节点名称
        rejected_node_name: 拒绝后转向的节点名称
        custom_interrupt_logic: 自定义中断逻辑函数
    """

    def human_approval(state: state_type) -> Command[Literal[approved_node_name, rejected_node_name]]:
        # 构建中断信息
        interrupt_info = {
            "question": "请审批此操作",
            "state_summary": str(state)
        }

        # 1. 添加固定的中断信息
        if interrupt_dict:
            interrupt_info.update(interrupt_dict)

        # 2. 如果有自定义逻辑，使用它来构建中断信息
        if custom_interrupt_logic:
            custom_info = custom_interrupt_logic(state)
            if isinstance(custom_info, dict):
                interrupt_info.update(custom_info)

        # 触发中断
        is_approved = interrupt(interrupt_info)

        # 处理审批结果
        if is_approved:
            print(f"✅ 操作已批准，转向节点: {approved_node_name}")
            return Command(goto=approved_node_name)
        else:
            print(f"❌ 操作被拒绝，转向节点: {rejected_node_name}")
            return Command(goto=rejected_node_name)

    return human_approval