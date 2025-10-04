"""
WebSocket 异步人工审批工具模块

提供通过 WebSocket 进行实时人工审批的工作流运行器
支持双向通信：服务器推送审批请求 → 客户端响应决策
"""

import uuid
import asyncio
import json
from typing import Optional, Dict, Any, List, Tuple, Callable
from langgraph.types import Command
from fastapi import WebSocket


class WebSocketApprovalManager:
    """WebSocket 审批管理器，用于在工作流中处理异步人工审批"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.pending_approval = None
        self.approval_response = None
        self.response_event = asyncio.Event()

    async def send_message(self, message_type: str, data: Any):
        """发送消息到客户端"""
        await self.websocket.send_json({
            "type": message_type,
            "data": data
        })

    async def request_approval(self, interrupt_info: Dict[str, Any]) -> bool:
        """
        请求人工审批

        Args:
            interrupt_info: 中断信息字典

        Returns:
            bool: True 表示批准，False 表示拒绝
        """
        # 重置事件
        self.response_event.clear()
        self.approval_response = None

        # 发送审批请求到客户端
        await self.send_message("approval_request", interrupt_info)

        # 等待客户端响应
        await self.response_event.wait()

        return self.approval_response

    def set_approval_response(self, is_approved: bool):
        """设置审批响应（由消息处理器调用）"""
        self.approval_response = is_approved
        self.response_event.set()


async def run_workflow_with_websocket_approval(
        graph,
        inputs: Dict[str, Any],
        websocket: WebSocket,
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        collect_messages: bool = True,
        on_message_callback: Optional[Callable] = None
) -> Tuple[Any, List[str], List[str]]:
    """
    通过 WebSocket 运行带人工审批的工作流

    Args:
        graph: 编译后的 LangGraph 对象
        inputs: 初始输入
        websocket: WebSocket 连接对象
        config: 配置字典（如果为 None，会自动生成 thread_id）
        debug: 是否打印调试信息
        collect_messages: 是否收集并发送消息
        on_message_callback: 消息回调函数（可选）

    Returns:
        tuple: (最终结果, agent响应列表, tool响应列表)
    """
    # 创建审批管理器
    approval_manager = WebSocketApprovalManager(websocket)

    # 如果没有提供 config，自动生成一个
    if config is None:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    agent_responses = []
    tool_responses = []

    # 发送开始消息
    await approval_manager.send_message("workflow_start", {
        "message": "工作流开始运行",
        "thread_id": config["configurable"]["thread_id"]
    })

    if debug:
        await approval_manager.send_message("debug", {"message": f"初始输入: {inputs}"})

    # 第一次执行 - 流式处理
    if collect_messages:
        stream = graph.stream(inputs, config=config, stream_mode="messages")

        for chunk in stream:
            message_chunk, metadata = chunk
            node_name = metadata.get('langgraph_node', 'unknown')

            if debug:
                await approval_manager.send_message("debug", {
                    "node": node_name,
                    "message": str(message_chunk)
                })

            # 收集并发送响应
            if hasattr(message_chunk, 'content') and message_chunk.content:
                if node_name == 'agent':
                    agent_responses.append(message_chunk.content)
                    await approval_manager.send_message("agent_message", {
                        "content": message_chunk.content
                    })
                    if on_message_callback:
                        await on_message_callback("agent", message_chunk.content)
                elif node_name == 'tools':
                    tool_responses.append(message_chunk.content)
                    await approval_manager.send_message("tool_message", {
                        "content": message_chunk.content
                    })
                    if on_message_callback:
                        await on_message_callback("tools", message_chunk.content)
    else:
        graph.invoke(inputs, config=config)

    # 循环处理所有中断
    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        iteration += 1

        # 获取当前状态
        state = graph.get_state(config)

        if debug:
            await approval_manager.send_message("debug", {
                "iteration": iteration,
                "next": state.next,
                "tasks_count": len(state.tasks) if state.tasks else 0
            })

        # 检查是否还有待执行的节点
        if not state.next:
            if debug:
                await approval_manager.send_message("debug", {"message": "工作流已完成"})
            break

        # 检查是否有中断任务
        has_interrupt = False
        interrupt_info = None

        if state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    has_interrupt = True
                    if task.interrupts:
                        interrupt_info = task.interrupts[0].value
                    break

        if not has_interrupt:
            # 没有中断，继续执行
            if debug:
                await approval_manager.send_message("debug", {"message": "继续执行下一个节点"})

            if collect_messages:
                stream = graph.stream(None, config=config, stream_mode="messages")
                for chunk in stream:
                    message_chunk, metadata = chunk
                    node_name = metadata.get('langgraph_node', 'unknown')

                    if debug:
                        await approval_manager.send_message("debug", {
                            "node": node_name,
                            "message": str(message_chunk)
                        })

                    if hasattr(message_chunk, 'content') and message_chunk.content:
                        if node_name == 'agent':
                            agent_responses.append(message_chunk.content)
                            await approval_manager.send_message("agent_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("agent", message_chunk.content)
                        elif node_name == 'tools':
                            tool_responses.append(message_chunk.content)
                            await approval_manager.send_message("tool_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("tools", message_chunk.content)
            else:
                graph.invoke(None, config=config)
            continue

        # 有中断，需要人工审批
        if debug:
            await approval_manager.send_message("debug", {
                "message": f"发现中断，需要人工审批（第 {iteration} 次）"
            })

        # 请求审批
        is_approved = await approval_manager.request_approval(interrupt_info)

        if is_approved:
            await approval_manager.send_message("approval_result", {
                "approved": True,
                "message": "审批通过，继续执行..."
            })

            if collect_messages:
                stream = graph.stream(Command(resume=True), config=config, stream_mode="messages")
                for chunk in stream:
                    message_chunk, metadata = chunk
                    node_name = metadata.get('langgraph_node', 'unknown')

                    if debug:
                        await approval_manager.send_message("debug", {
                            "node": node_name,
                            "message": str(message_chunk)
                        })

                    if hasattr(message_chunk, 'content') and message_chunk.content:
                        if node_name == 'agent':
                            agent_responses.append(message_chunk.content)
                            await approval_manager.send_message("agent_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("agent", message_chunk.content)
                        elif node_name == 'tools':
                            tool_responses.append(message_chunk.content)
                            await approval_manager.send_message("tool_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("tools", message_chunk.content)
            else:
                graph.invoke(Command(resume=True), config=config)
        else:
            await approval_manager.send_message("approval_result", {
                "approved": False,
                "message": "审批被拒绝"
            })

            if collect_messages:
                stream = graph.stream(Command(resume=False), config=config, stream_mode="messages")
                for chunk in stream:
                    message_chunk, metadata = chunk
                    node_name = metadata.get('langgraph_node', 'unknown')

                    if debug:
                        await approval_manager.send_message("debug", {
                            "node": node_name,
                            "message": str(message_chunk)
                        })

                    if hasattr(message_chunk, 'content') and message_chunk.content:
                        if node_name == 'agent':
                            agent_responses.append(message_chunk.content)
                            await approval_manager.send_message("agent_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("agent", message_chunk.content)
                        elif node_name == 'tools':
                            tool_responses.append(message_chunk.content)
                            await approval_manager.send_message("tool_message", {
                                "content": message_chunk.content
                            })
                            if on_message_callback:
                                await on_message_callback("tools", message_chunk.content)
            else:
                graph.invoke(Command(resume=False), config=config)

    if iteration >= max_iterations:
        await approval_manager.send_message("warning", {
            "message": f"达到最大迭代次数 {max_iterations}"
        })

    final_state = graph.get_state(config)

    # 发送完成消息
    await approval_manager.send_message("workflow_complete", {
        "message": "工作流执行完成",
        "agent_responses_count": len(agent_responses),
        "tool_responses_count": len(tool_responses),
        "iterations": iteration
    })

    if debug:
        await approval_manager.send_message("debug", {
            "final_state": str(final_state.values),
            "agent_responses_count": len(agent_responses),
            "tool_responses_count": len(tool_responses),
            "total_iterations": iteration
        })

    return final_state.values, agent_responses, tool_responses, approval_manager
