from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState


def create_chat_node(model,system_prompt):
    """
    创建一个通用的工具调用节点
    :param tools_dict: 工具字典，格式为 {tool_name: tool_instance}
    """

    def call_model(
            state: AgentState,
            config: RunnableConfig,
    ):
        # 定义prompt提示词
        prompt = system_prompt
        response = model.invoke([prompt] + state["messages"], config)
        # 需要返回一个messages列表
        return {"messages": [response]}
    return call_model

