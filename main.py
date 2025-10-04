import uuid
import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# WebSocket 端点：运行带人工审批的 Agent 工作流
@app.websocket("/ws/agent/knowledge-graph")
async def websocket_knowledge_graph_agent(websocket: WebSocket):
    """
    WebSocket 端点 - 知识图谱 Agent（带人工审批）

    消息协议：
    客户端 → 服务器：
        - {"type": "start", "data": {"user_input": "..."}} - 开始工作流
        - {"type": "approval_response", "data": {"approved": true/false}} - 审批响应

    服务器 → 客户端：
        - {"type": "workflow_start", "data": {...}} - 工作流开始
        - {"type": "agent_message", "data": {"content": "..."}} - Agent 响应
        - {"type": "tool_message", "data": {"content": "..."}} - 工具响应
        - {"type": "approval_request", "data": {...}} - 请求人工审批
        - {"type": "approval_result", "data": {...}} - 审批结果
        - {"type": "workflow_complete", "data": {...}} - 工作流完成
        - {"type": "debug", "data": {...}} - 调试信息
        - {"type": "error", "data": {"message": "..."}} - 错误信息
    """
    await websocket.accept()

    try:
        from Agents.KnowledgeGraphAgent.StructuredDataAgent.UserIntentAgent.UserIntentAgent import graph
        from utils.WebSocketApproval import run_workflow_with_websocket_approval

        approval_manager = None

        # 消息处理循环
        while True:
            # 接收客户端消息
            message = await websocket.receive_json()
            message_type = message.get("type")
            data = message.get("data", {})

            if message_type == "start":
                # 启动工作流
                user_input = data.get("user_input", "")
                debug = data.get("debug", False)

                if not user_input:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "user_input 不能为空"}
                    })
                    continue

                # 配置
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                inputs = {"messages": [("user", user_input)]}

                # 在后台任务中运行工作流
                async def run_workflow():
                    try:
                        result, agent_msgs, tool_msgs, mgr = await run_workflow_with_websocket_approval(
                            graph=graph,
                            inputs=inputs,
                            websocket=websocket,
                            config=config,
                            debug=debug,
                            collect_messages=True
                        )
                        # 保存 approval_manager 引用
                        nonlocal approval_manager
                        approval_manager = mgr
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": f"工作流执行错误: {str(e)}"}
                        })

                # 启动异步任务
                asyncio.create_task(run_workflow())

            elif message_type == "approval_response":
                # 处理审批响应
                if approval_manager is None:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "没有正在等待的审批请求"}
                    })
                    continue

                is_approved = data.get("approved", False)
                approval_manager.set_approval_response(is_approved)

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"未知的消息类型: {message_type}"}
                })

    except WebSocketDisconnect:
        print("WebSocket 连接已断开")
    except Exception as e:
        print(f"WebSocket 错误: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8066,workers=1)