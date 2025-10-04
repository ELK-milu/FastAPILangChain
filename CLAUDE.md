# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FastAPI + LangChain AI Agent 开发项目，专注于使用 DeepSeek 模型（通过 SiliconFlow API）创建智能 Agent。项目展示了多种 Agent 架构模式，包括 ReAct Agent、多 Agent 协作、知识图谱 Agent 等。

## Common Commands

### 运行 FastAPI 应用
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 测试 API 端点
使用 `test_main.http` 文件进行 API 测试（支持 VS Code REST Client 插件）：
- `GET http://127.0.0.1:8000/`
- `GET http://127.0.0.1:8000/hello/{name}`

### 测试 WebSocket Agent（推荐）
访问浏览器测试页面：
```
http://127.0.0.1:8000/test-agent
```
提供可视化界面测试带人工审批的 Knowledge Graph Agent

### 运行独立 Agent 示例
```bash
# 运行知识图谱 Agent 测试
python test_knowledge_graph_agent.py

# 运行其他 Agent 示例（在 Agents/ 目录中）
python -m Agents.KnowledgeGraphAgent.Agent
```

## Architecture Overview

### Core Technologies Stack
- **FastAPI** - REST API 框架
- **LangChain** - AI Agent 编排框架
- **LangGraph** - 状态图管理（StateGraph 模式）
- **DeepSeek-V3** - LLM（通过 SiliconFlow API）
- **Neo4j** - 图数据库（用于知识图谱 Agent）

### Directory Structure
```
├── main.py                    # FastAPI 入口点（包含 WebSocket 端点）
├── utils/                     # 通用工具模块
│   ├── env_utils.py          # 环境变量加载（.env 配置）
│   ├── ChatNode.py           # LangGraph Chat 节点工厂
│   ├── ToolNode.py           # LangGraph Tool 节点工厂（支持人工审批）
│   ├── HumanApproval.py      # 通用人工审批节点
│   ├── ConditionNode.py      # 条件路由节点（判断是否继续）
│   ├── OutputParser.py       # 流式输出解析器和工作流运行器（命令行版）
│   └── WebSocketApproval.py  # WebSocket 异步审批工作流运行器
├── Agents/                    # 生产级 Agent 实现
│   └── KnowledgeGraphAgent/  # 知识图谱 Agent（Neo4j 集成）
└── test_*.py                  # 测试文件
```

### 关键架构模式

#### 1. LangGraph StateGraph 模式
所有 Agent 都基于 LangGraph 的 StateGraph 构建：
- **节点（Node）**：执行特定功能的函数（chat_node, tool_node 等）
- **边（Edge）**：节点间的转换关系（条件边和固定边）
- **状态（State）**：使用 `AgentState` 类型管理消息历史

#### 2. 工具调用流程（utils/ToolNode.py）
- **基础工具节点**：`create_tool_node()` - 简单的工具执行
- **人工审批工具节点**：`create_tool_node_with_approval()` - 支持执行前/后审批
  - `approval_strategy="before"` - 工具执行前需要人工确认
  - `approval_strategy="after"` - 工具执行后需要人工确认结果
  - `should_approve_fn` - 自定义审批条件（例如只审批危险操作）
  - `custom_interrupt_logic` - 自定义中断信息展示

#### 3. 人工审批模式（Human-in-the-Loop）
使用 `interrupt()` 和 `Command` 实现人工介入：
```python
# 触发中断，等待人工审批
is_approved = interrupt(interrupt_info)

# 根据审批结果路由
if is_approved:
    return Command(goto="approved_node_name")
else:
    return Command(goto="rejected_node_name")
```

#### 4. 工作流运行器
**命令行版本（utils/OutputParser.py）**：
- **`run_workflow_with_approval()`** - 自动处理所有中断的工作流运行器
- **`run_workflow_with_approval_streaming()`** - 流式版本，实时显示消息
- 支持 `auto_approve=True` 用于测试自动批准所有审批
- 使用 `input()` 阻塞式交互，仅适合命令行环境

**WebSocket 版本（utils/WebSocketApproval.py）**：
- **`run_workflow_with_websocket_approval()`** - 通过 WebSocket 进行异步人工审批
- 双向实时通信：服务器推送审批请求 → 客户端响应决策
- 支持流式输出、调试模式、自定义消息回调
- **推荐用于生产环境和 API 集成**

#### 5. WebSocket API 集成（main.py）
FastAPI WebSocket 端点实现人工审批工作流：
- **端点**: `ws://127.0.0.1:8000/ws/agent/knowledge-graph`
- **测试页面**: `http://127.0.0.1:8000/test-agent`

**消息协议**：
```javascript
// 客户端 → 服务器
{"type": "start", "data": {"user_input": "...", "debug": false}}
{"type": "approval_response", "data": {"approved": true/false}}

// 服务器 → 客户端
{"type": "workflow_start", "data": {...}}
{"type": "agent_message", "data": {"content": "..."}}
{"type": "tool_message", "data": {"content": "..."}}
{"type": "approval_request", "data": {...}}  // 请求人工审批
{"type": "approval_result", "data": {...}}
{"type": "workflow_complete", "data": {...}}
{"type": "debug", "data": {...}}
{"type": "error", "data": {"message": "..."}}
```

### Environment Configuration

必需的环境变量（在 `.env` 文件中配置）：
```bash
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

通过 `utils/env_utils.py` 加载环境变量：
```python
from utils.env_utils import SILICONFLOW_API_KEY, NEO4J_URI
```

### Model Integration Pattern

标准的模型初始化模式（使用 SiliconFlow 作为 OpenAI 兼容端点）：
```python
from langchain_openai import ChatOpenAI
from utils.env_utils import SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY

model = ChatOpenAI(
    base_url=SILICONFLOW_BASE_URL,
    api_key=SILICONFLOW_API_KEY,
    model="deepseek-v3",
    temperature=0
)
```

### Utility Functions Overview

#### `utils/ChatNode.py`
创建标准的 LangGraph chat 节点，接受 model 和 system_prompt。

#### `utils/ToolNode.py`
- `create_tool_node(tools)` - 基础工具执行节点
- `create_tool_node_with_approval(tools, approved_node_name, rejected_node_name, ...)` - 支持人工审批的工具节点

#### `utils/HumanApproval.py`
创建独立的人工审批节点，可用于任意状态类型。

#### `utils/ConditionNode.py`
`should_continue(state)` - 检查最后一条消息是否包含工具调用，决定是否继续执行。

#### `utils/OutputParser.py`
- `agent_with_tool_stream_parser()` - 解析流式消息并分类（agent 响应 vs tool 响应）
- `run_workflow_with_approval()` - 自动化处理所有人工审批的工作流运行器（命令行版）
- `run_workflow_with_approval_streaming()` - 流式版本，实时显示输出（命令行版）

#### `utils/WebSocketApproval.py`
- `WebSocketApprovalManager` - WebSocket 审批管理器类
- `run_workflow_with_websocket_approval()` - 异步 WebSocket 工作流运行器
- 支持双向实时通信、流式消息推送、自定义回调

### Development Notes

#### 项目不含 requirements.txt
关键依赖包括：
- `fastapi`, `uvicorn[standard]`, `websockets`
- `langchain`, `langchain-openai`, `langchain-neo4j`
- `langgraph`
- `pydantic`, `python-dotenv`

#### 代码风格
- 中文注释和文档字符串
- 所有 Agent 使用 LangGraph StateGraph 模式
- Pydantic 模型用于结构化数据验证
- 工具函数使用 `@tool` 装饰器定义

#### 安全注意事项
- API 密钥必须通过环境变量管理，不要硬编码
- Neo4j 凭证同样使用环境变量
- 部分示例文件可能包含旧的硬编码密钥（仅用于学习，生产环境需移除）