from langgraph.prebuilt.chat_agent_executor import AgentState


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果无工具调用则结束
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"