

def agent_with_tool_stream_parser(stream,agent_responses=[],tool_responses=[],debug=False):
    for chunk in stream:
        message_chunk, metadata = chunk
        node_name = metadata.get('langgraph_node', 'unknown')
        if debug:
            print(message_chunk)
        if hasattr(message_chunk, 'content') and message_chunk.content:
            if node_name == 'agent':
                agent_responses.append(message_chunk.content)
            elif node_name == 'tools':
                tool_responses.append(message_chunk.content)
    return agent_responses,tool_responses


import uuid
from typing import Optional, Dict, Any, List, Tuple
from langgraph.types import Command


def run_workflow_with_approval(
        graph,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        auto_approve: bool = False
) -> Tuple[Any, List[str], List[str]]:
    """
    运行带审批的工作流，自动处理所有 interrupt

    Args:
        graph: 编译后的 LangGraph 对象
        inputs: 初始输入
        config: 配置字典（如果为 None，会自动生成 thread_id）
        debug: 是否打印调试信息
        auto_approve: 是否自动批准所有审批（用于测试）

    Returns:
        tuple: (最终结果, agent响应列表, tool响应列表)
    """
    # 如果没有提供 config，自动生成一个
    if config is None:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent_responses = []
    tool_responses = []

    print("=" * 60)
    print("🚀 开始运行工作流")
    print("=" * 60)

    # 第一次执行
    if debug:
        print(f"\n📥 初始输入: {inputs}")

    result = graph.invoke(inputs, config=config)

    # 循环处理所有中断
    iteration = 0
    max_iterations = 100  # 防止无限循环

    while iteration < max_iterations:
        iteration += 1

        # 获取当前状态
        state = graph.get_state(config)

        if debug:
            print(f"\n🔍 迭代 {iteration}:")
            print(f"   next: {state.next}")
            print(f"   tasks数量: {len(state.tasks) if state.tasks else 0}")

        # 检查是否还有待执行的节点
        if not state.next:
            # 没有下一个节点，工作流已完成
            if debug:
                print("   ✅ 工作流已完成，没有待执行节点")
            break

        # 检查是否有中断任务
        has_interrupt = False
        interrupt_info = None

        if state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    has_interrupt = True
                    # 获取第一个中断的信息
                    if task.interrupts:
                        interrupt_info = task.interrupts[0].value
                    if debug:
                        print(f"   ⏸️  发现中断任务: {task.name}")
                    break

        if not has_interrupt:
            # 没有中断，但还有待执行节点，继续执行
            if debug:
                print("   ▶️  没有中断，继续执行下一个节点")
            result = graph.invoke(None, config=config)
            continue

        # 有中断，需要人工审批
        print("\n" + "-" * 60)
        print(f"⏸️  工作流需要人工审批 (第 {iteration} 次)")
        print("-" * 60)

        # 显示中断信息
        if interrupt_info:
            print(f"\n📋 审批请求:")
            print(f"   问题: {interrupt_info.get('question', '需要审批')}")

            # 显示工具调用信息
            if 'tool_calls' in interrupt_info:
                print(f"\n🔧 待执行的工具调用:")
                for i, tc in enumerate(interrupt_info['tool_calls'], 1):
                    print(f"   {i}. 工具名称: {tc.get('tool_name', 'unknown')}")
                    args = tc.get('arguments', {})
                    print(f"      参数:")
                    for key, value in args.items():
                        # 如果值太长，截断显示
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"        - {key}: {value_str}")

            # 显示工具结果信息
            if 'tool_results' in interrupt_info:
                print(f"\n📊 工具执行结果:")
                for i, tr in enumerate(interrupt_info['tool_results'], 1):
                    print(f"   {i}. 工具名称: {tr.get('tool_name', 'unknown')}")
                    result_str = str(tr.get('result', 'N/A'))
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    print(f"      结果: {result_str}")

            # 显示其他自定义信息
            other_info = {k: v for k, v in interrupt_info.items()
                          if k not in ['question', 'tool_calls', 'tool_results', 'state_summary']}
            if other_info:
                print(f"\n📝 其他信息:")
                for key, value in other_info.items():
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    print(f"   {key}: {value_str}")

            if debug and 'state_summary' in interrupt_info:
                print(f"\n🔍 状态摘要: {interrupt_info['state_summary']}")

        # 获取用户输入或自动批准
        print("\n" + "-" * 60)
        if auto_approve:
            is_approved = True
            print("🤖 自动批准模式: 已批准")
        else:
            user_input = input("❓ 是否批准? (y/n): ").strip().lower()
            is_approved = user_input in ['y', 'yes', '是']

        if is_approved:
            print("✅ 审批通过，继续执行...")
            result = graph.invoke(Command(resume=True), config=config)
        else:
            print("❌ 审批被拒绝，执行拒绝逻辑...")
            result = graph.invoke(Command(resume=False), config=config)

        print("-" * 60 + "\n")

    if iteration >= max_iterations:
        print(f"⚠️  警告: 达到最大迭代次数 {max_iterations}，可能存在无限循环")

    print("\n" + "=" * 60)
    print("✅ 工作流执行完成")
    print("=" * 60)

    if debug:
        print(f"\n📤 最终结果: {result}")
        print(f"   总迭代次数: {iteration}")

    return result, agent_responses, tool_responses


def _process_stream_messages(stream, agent_responses: List[str], tool_responses: List[str], debug: bool):
    """处理流式消息的辅助函数"""
    for chunk in stream:
        message_chunk, metadata = chunk
        node_name = metadata.get('langgraph_node', 'unknown')

        if debug:
            print(f"[DEBUG][{node_name}] {message_chunk}")

        if hasattr(message_chunk, 'content') and message_chunk.content:
            if node_name == 'agent':
                agent_responses.append(message_chunk.content)
                if debug:
                    print(f"🤖 Agent: {message_chunk.content}")
            elif node_name == 'tools':
                tool_responses.append(message_chunk.content)
                if debug:
                    print(f"🔧 Tool: {message_chunk.content}")


def _display_approval_request(interrupt_info: Dict[str, Any]):
    """显示审批请求信息"""
    print(f"\n📋 审批请求:")
    print(f"   问题: {interrupt_info.get('question', '需要审批')}")

    if 'tool_calls' in interrupt_info:
        print(f"\n🔧 待执行的工具调用:")
        for i, tc in enumerate(interrupt_info['tool_calls'], 1):
            print(f"   {i}. 工具: {tc.get('tool_name', 'unknown')}")
            args = tc.get('arguments', {})
            for key, value in args.items():
                value_str = str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                print(f"      - {key}: {value_str}")

    if 'tool_results' in interrupt_info:
        print(f"\n📊 工具执行结果:")
        for i, tr in enumerate(interrupt_info['tool_results'], 1):
            print(f"   {i}. 工具: {tr.get('tool_name', 'unknown')}")
            result_str = str(tr.get('result', 'N/A'))[:100] + ("..." if len(str(tr.get('result', 'N/A'))) > 100 else "")
            print(f"      结果: {result_str}")

    other_info = {k: v for k, v in interrupt_info.items()
                  if k not in ['question', 'tool_calls', 'tool_results', 'state_summary']}
    if other_info:
        print(f"\n📝 其他信息:")
        for key, value in other_info.items():
            value_str = str(value)[:200] + ("..." if len(str(value)) > 200 else "")
            print(f"   {key}: {value_str}")


def _get_user_approval(auto_approve: bool) -> bool:
    """获取用户审批决策"""
    print("\n" + "-" * 60)
    if auto_approve:
        print("🤖 自动批准模式: 已批准")
        return True
    else:
        user_input = input("❓ 是否批准? (y/n): ").strip().lower()
        return user_input in ['y', 'yes', '是']


def _check_interrupt(state) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """检查是否有中断任务"""
    if not state.tasks:
        return False, None

    for task in state.tasks:
        if hasattr(task, 'interrupts') and task.interrupts:
            return True, task.interrupts[0].value
    return False, None


def run_workflow_with_approval_streaming(
        graph,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        auto_approve: bool = False,
        collect_messages: bool = True
) -> Tuple[Any, List[str], List[str]]:
    """
    运行带审批的工作流（流式版本），自动处理所有 interrupt，并收集消息

    Args:
        graph: 编译后的 LangGraph 对象
        inputs: 初始输入
        config: 配置字典
        debug: 是否打印调试信息
        auto_approve: 是否自动批准所有审批
        collect_messages: 是否收集并显示消息

    Returns:
        tuple: (最终结果, agent响应列表, tool响应列表)
    """
    if config is None:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent_responses = []
    tool_responses = []

    print("=" * 60)
    print("🚀 开始运行工作流（流式模式）")
    print("=" * 60)

    if debug:
        print(f"\n📥 初始输入: {inputs}")

    # 第一次执行
    if collect_messages:
        print("\n📨 开始收集消息...")
        stream = graph.stream(inputs, config=config, stream_mode="messages")
        _process_stream_messages(stream, agent_responses, tool_responses, debug)
    else:
        graph.invoke(inputs, config=config)

    # 循环处理所有中断
    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        iteration += 1
        state = graph.get_state(config)

        if debug:
            print(f"\n🔍 迭代 {iteration}: next={state.next}, tasks={len(state.tasks) if state.tasks else 0}")

        # 检查是否完成
        if not state.next:
            if debug:
                print("   ✅ 工作流已完成")
            break

        # 检查中断
        has_interrupt, interrupt_info = _check_interrupt(state)

        if not has_interrupt:
            if debug:
                print("   ▶️  继续执行")

            if collect_messages:
                stream = graph.stream(None, config=config, stream_mode="messages")
                _process_stream_messages(stream, agent_responses, tool_responses, debug)
            else:
                graph.invoke(None, config=config)
            continue

        # 处理人工审批
        print("\n" + "-" * 60)
        print(f"⏸️  工作流需要人工审批 (第 {iteration} 次)")
        print("-" * 60)

        if interrupt_info:
            _display_approval_request(interrupt_info)

        is_approved = _get_user_approval(auto_approve)
        resume_command = Command(resume=is_approved)

        print(f"{'✅ 审批通过' if is_approved else '❌ 审批被拒绝'}，继续执行...")

        if collect_messages:
            stream = graph.stream(resume_command, config=config, stream_mode="messages")
            _process_stream_messages(stream, agent_responses, tool_responses, debug)
        else:
            graph.invoke(resume_command, config=config)

        print("-" * 60 + "\n")

    if iteration >= max_iterations:
        print(f"⚠️  警告: 达到最大迭代次数 {max_iterations}")

    final_state = graph.get_state(config)

    print("\n" + "=" * 60)
    print("✅ 工作流执行完成")
    print("=" * 60)

    if debug:
        print(f"\n📤 最终状态: {final_state.values}")
        print(f"   Agent响应数: {len(agent_responses)}")
        print(f"   Tool响应数: {len(tool_responses)}")
        print(f"   总迭代次数: {iteration}")

    return final_state.values, agent_responses, tool_responses


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    使用示例 - 替换你原来的代码
    """

    # 原来的代码:
    # graph = workflow.compile(checkpointer)
    # config = {"configurable": {"thread_id": uuid.uuid4()}}
    # inputs = {"messages": [("user", "I'd like a BOM graph...")]}
    # result = graph.invoke(inputs, config=config)
    # bool_input = input("Approve? (y/n): ")
    # if bool_input == "y":
    #     graph.invoke(Command(resume=True), config=config)
    # else:
    #     graph.invoke(Command(resume=False), config=config)

    # 新的代码（简单版）:
    # graph = workflow.compile(checkpointer)
    # config = {"configurable": {"thread_id": uuid.uuid4()}}
    # inputs = {"messages": [("user", "I'd like a BOM graph...")]}
    #
    # result, agent_msgs, tool_msgs = run_workflow_with_approval(
    #     graph=graph,
    #     config=config,
    #     inputs=inputs,
    #     debug=False
    # )
    #
    # print(f"\n📊 执行统计:")
    # print(f"   Agent消息数: {len(agent_msgs)}")
    # print(f"   Tool消息数: {len(tool_msgs)}")

    # 流式版本（可以看到实时输出）:
    # result, agent_msgs, tool_msgs = run_workflow_with_streaming(
    #     graph=graph,
    #     config=config,
    #     inputs=inputs,
    #     debug=False,
    #     collect_messages=True
    # )

    pass