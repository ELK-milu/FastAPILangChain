"""
测试转换后的 Knowledge Graph Agent
"""

from Agents.KnowledgeGraphAgent.Agent import user_intent_agent, KnowledgeGraphState
from langchain_core.messages import HumanMessage


def test_knowledge_graph_agent():
    """测试知识图谱代理的基本功能"""

    # 创建初始状态
    initial_state: KnowledgeGraphState = {
        "messages": [HumanMessage(content="我想创建一个社交网络图谱，分析朋友之间的关系")],
        "perceived_user_goal": None,
        "approved_user_goal": None
    }

    print("=== 开始测试 Knowledge Graph Agent ===")
    print(f"用户输入: {initial_state['messages'][0].content}")
    print("\n执行代理...")

    try:
        # 运行代理
        result = user_intent_agent.invoke(initial_state)

        print("\n=== 代理响应 ===")
        for i, message in enumerate(result["messages"]):
            print(f"消息 {i+1}: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"工具调用: {message.tool_calls}")

        print(f"\n=== 状态信息 ===")
        print(f"感知的用户目标: {result.get('perceived_user_goal')}")
        print(f"已批准的用户目标: {result.get('approved_user_goal')}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def test_approval_workflow():
    """测试批准工作流程"""

    print("\n\n=== 测试批准工作流程 ===")

    # 第一步：用户提出需求
    state1: KnowledgeGraphState = {
        "messages": [HumanMessage(content="我想分析电商平台的用户购买行为模式")],
        "perceived_user_goal": None,
        "approved_user_goal": None
    }

    result1 = user_intent_agent.invoke(state1)
    print("第一轮对话完成")

    # 第二步：用户确认目标
    if result1.get("perceived_user_goal"):
        state2: KnowledgeGraphState = {
            "messages": result1["messages"] + [HumanMessage(content="是的，这正是我想要的！")],
            "perceived_user_goal": result1.get("perceived_user_goal"),
            "approved_user_goal": result1.get("approved_user_goal")
        }

        result2 = user_intent_agent.invoke(state2)
        print("第二轮对话完成")
        print(f"最终批准的目标: {result2.get('approved_user_goal')}")


if __name__ == "__main__":
    test_knowledge_graph_agent()
    test_approval_workflow()