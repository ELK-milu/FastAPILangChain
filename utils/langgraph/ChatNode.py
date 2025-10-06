from typing import Callable

from langchain_core.runnables import RunnableConfig


def create_chat_node(model, system_prompt):
    """
    创建一个通用的 chat 节点

    重要：必须返回所有需要保留的自定义字段！
    Reducer 不会自动保留未返回的字段，它只负责合并返回的值。

    注意：不使用类型注解以支持自定义 State 类型

    :param model: 聊天模型
    :param system_prompt: 系统提示词
    """

    def call_model(state, config):

        # 定义prompt提示词
        prompt = system_prompt
        response = model.invoke([prompt] + state["messages"], config)

        # 构建返回值：包含 messages 和所有自定义字段
        result = {"messages": [response]}

        # LangGraph 管理字段列表（不由节点返回）
        MANAGED_FIELDS = {"remaining_steps", "is_last_step", "messages"}

        # 返回所有自定义字段，确保它们在节点间传递
        for key in state.keys():
            if key not in MANAGED_FIELDS:
                result[key] = state[key]

        return result

    return call_model


def create_chat_node_inject(model, inject_system_prompt:Callable):
    """
    创建一个通用的 chat 节点

    重要：必须返回所有需要保留的自定义字段！
    Reducer 不会自动保留未返回的字段，它只负责合并返回的值。

    注意：不使用类型注解以支持自定义 State 类型

    :param model: 聊天模型
    :param system_prompt: 系统提示词
    """

    def call_model(state, config):

        # 定义prompt提示词
        prompt = inject_system_prompt(state)
        response = model.invoke([prompt] + state["messages"], config)

        # 构建返回值：包含 messages 和所有自定义字段
        result = {"messages": [response]}

        # LangGraph 管理字段列表（不由节点返回）
        MANAGED_FIELDS = {"remaining_steps", "is_last_step", "messages"}

        # 返回所有自定义字段，确保它们在节点间传递
        for key in state.keys():
            if key not in MANAGED_FIELDS:
                result[key] = state[key]

        return result

    return call_model
