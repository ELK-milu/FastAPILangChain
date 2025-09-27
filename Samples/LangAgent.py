from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt.chat_agent_executor import AgentState
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from IPython.display import Image, display


class LangAgent:
    def __init__(self,model,tools,prompt,call_func=None):
        self.tools : list[BaseTool] = tools
        self.model : BaseChatModel = model
        self.prompt = prompt
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model = model.bind_tools(self.tools)
        self.call_func : callable = call_func

    def tool_node(self,state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def call_model(
            self,
            state: AgentState,
            config: RunnableConfig,
    ):
        if self.call_func is not None:
            response =  self.call_func(state,config)
        else:
            response = self.model.invoke([self.prompt] + state["messages"], config)
        # 需要返回一个messages列表
        return {"messages": [response]}

