import uuid
from typing import TypedDict, Annotated, Sequence
from operator import add

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils.langgraph.ChatNode import create_chat_node, create_chat_node_inject
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.models import DeepSeek_V3


# 假设的模型导入（根据你的实际情况调整）
# from utils.models import DeepSeek_V3

# 1. 正确定义 State - 不继承 AgentState
class GameState(AgentState):
    player_name: str
    score: int
    level: int
    inventory: list[str]


@tool(description="增加玩家分数")
@needs_state
def add_score_tool(points: int,state) -> str:
    """增加玩家分数

    Args:
        points: 要增加的分数
    """
    current_score = state.get("score",50)
    state["score"] = current_score + points

    return f"成功增加 {points} 分"


@tool(description="提升玩家等级")
@needs_state
def level_up_tool(levels: int,state) -> str:
    """提升玩家等级

    Args:
        levels: 要升级的等级数，默认1级
    """

    current_level = state.get("level",1)
    state["level"] = current_level + levels

    return f"成功升级 {levels} 级"


@tool(description="添加物品到背包")
@needs_state
def add_item_tool(item_name: str,state) -> str:
    """添加物品到背包

    Args:
        item_name: 要添加的物品名称
    """

    inventory = state.get("inventory",[])
    state["inventory"].append(item_name)

    return f"成功添加物品: {item_name}"


tools = [add_score_tool, level_up_tool, add_item_tool]


def create_system_message(state):
    system_message = f"""你是一个游戏助手。当前玩家状态：
    - 玩家名称：{state['player_name']}
    - 分数：{state['score']}
    - 等级：{state['level']}
    - 背包：{state['inventory']}
    
    根据用户请求调用相应的工具：
    - 增加分数时调用 add_score_tool
    - 升级时调用 level_up_tool
    - 添加物品时调用 add_item_tool
    
    每次调用完毕后，将玩家状态打印出来
    """
    return system_message

model = DeepSeek_V3.bind_tools(tools)  # 使用你的模型
call_chat_node = create_chat_node_inject(model, create_system_message)


# 5. 条件判断函数
def should_continue(state: GameState):
    """判断是否继续调用工具"""
    last_message = state['messages'][-1]

    # 如果有工具调用，继续
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # 否则结束
    return "end"


tools_node = create_tool_node_with_state(tools)

# 6. 构建图
def create_game_graph():
    workflow = StateGraph(GameState)

    # 添加节点
    workflow.add_node("agent", call_chat_node)
    workflow.add_node("tools", tools_node)

    # 设置入口
    workflow.set_entry_point("agent")

    # 添加条件边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # 工具执行后返回 agent
    workflow.add_edge("tools", "agent")

    # 使用 checkpointer
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer,debug=True)


# 7. 使用示例
def main():
    game_graph = create_game_graph()

    # 创建唯一的线程 ID
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 初始状态
    initial_state = {
        "messages": [],
        "player_name": "小明",
        "score": 100,
        "level": 1,
        "inventory": ["初始剑", "药水"]
    }

    print("=== 初始状态 ===")
    print(f"玩家: {initial_state['player_name']}")
    print(f"分数: {initial_state['score']}")
    print(f"等级: {initial_state['level']}")
    print(f"背包: {initial_state['inventory']}")
    print()

    # 场景1: 增加分数
    print("=== 场景1: 增加分数 ===")
    state1 = {
        **initial_state,
        "messages": [HumanMessage(content="帮我增加10分")]
    }
    result1 = game_graph.invoke(state1, config=config)
    initial_state = result1
    print(f"initial_state: {initial_state}")
    print()

    # 场景2: 升级（使用 checkpointer 自动恢复之前的状态）
    print("=== 场景2: 升级 ===")
    state2 = {"messages": [HumanMessage(content="帮我升1级")]}
    result2 = game_graph.invoke(state2, config=config)
    initial_state = result2
    print(f"initial_state: {initial_state}")

    print()

    # 场景3: 添加物品
    print("=== 场景3: 添加物品 ===")
    state3 = {"messages": [HumanMessage(content="我获得了'金属盾牌'")]}
    result3 = game_graph.invoke(state3, config=config)
    initial_state = result3
    print(f"initial_state: {initial_state}")
    print()

    # 最终状态
    print("=== 最终状态 ===")
    print(f"玩家: {result3['player_name']}")
    print(f"分数: {result3['score']}")
    print(f"等级: {result3['level']}")
    print(f"背包: {result3['inventory']}")


if __name__ == "__main__":
    main()