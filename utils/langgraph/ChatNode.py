from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState


def create_chat_node(model, system_prompt):
    """
    创建一个通用的 chat 节点

    注意：节点返回值会与 state 合并，只返回需要更新的字段即可
    LangGraph 会自动保留未返回的字段

    :param model: 聊天模型
    :param system_prompt: 系统提示词
    """

    def call_model(
            state: AgentState,
            config: RunnableConfig,
    ):
        print("执行chat_node")
        print(f"[DEBUG ChatNode] 接收到的 state keys: {list(state.keys())}")
        print(f"[DEBUG ChatNode] state 内容: {dict(state)}")

        # 定义prompt提示词
        prompt = system_prompt
        response = model.invoke([prompt] + state["messages"], config)

        # 只返回需要更新的字段
        # LangGraph 会自动保留 state 中的其他字段（如 test_num）
        result = {"messages": [response]}
        print(f"[DEBUG ChatNode] 返回值: {result}")
        return result

    return call_model

