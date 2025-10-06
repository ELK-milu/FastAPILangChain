# Agent 封装为工具模式指南

本文档展示如何将一个 LangGraph Agent 封装为工具，供其他 Agent 调用。

## 核心概念

将 Agent 封装为工具的本质是：
1. **编译 Agent 的 StateGraph** 得到可调用的 `graph` 对象
2. **创建一个 `@tool` 装饰的函数**，在函数内部调用 `graph.invoke()` 或 `graph.stream()`
3. **处理状态传递**：通过 `@needs_state` 在工具间共享状态

## 方法一：简单封装（无状态共享）

适用于子 Agent 独立运行，不需要访问父 Agent 状态的场景。

```python
import uuid
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# ========== 子 Agent 定义 ==========
from langgraph.prebuilt.chat_agent_executor import AgentState

# 假设你已经有一个编译好的子 Agent graph
# 例如：FileSuggestionAgent
from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import graph as file_suggestion_graph

@tool(description="调用文件建议 Agent 来获取推荐的导入文件")
def call_file_suggestion_agent(user_query: str) -> dict:
    """
    调用文件建议 Agent 来获取推荐的导入文件。

    Args:
        user_query: 用户查询，描述需要什么样的文件

    Returns:
        dict: 包含 'status' 和 'result' 的字典
    """
    # 为子 Agent 创建独立的配置
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())  # 每次调用使用新的 thread
        }
    }

    # 构造输入
    inputs = {
        "messages": [("user", user_query)]
    }

    try:
        # 调用子 Agent
        result = file_suggestion_graph.invoke(inputs, config=config)

        # 提取最后的消息作为结果
        last_message = result["messages"][-1]

        return {
            "status": "success",
            "result": last_message.content if hasattr(last_message, 'content') else str(last_message)
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"调用文件建议 Agent 失败: {str(e)}"
        }
```

## 方法二：状态共享封装（推荐）

适用于子 Agent 需要访问或修改父 Agent 状态的场景。

```python
from langchain_core.tools import tool
from utils.langgraph.ToolNode import needs_state
from langgraph.prebuilt.chat_agent_executor import AgentState

# ========== 定义父 Agent 的状态 ==========
class ParentAgentState(AgentState):
    # 自定义字段
    approved_files: list[str]  # 子 Agent 返回的文件列表
    user_goal: str             # 用户目标

# ========== 封装子 Agent 为工具 ==========
@tool(description="调用文件建议 Agent 并将结果存储到父 Agent 状态")
@needs_state
def call_file_suggestion_agent_with_state(
    user_query: str,
    state: ParentAgentState = None
) -> dict:
    """
    调用文件建议 Agent 并将结果存储到父 Agent 状态。

    Args:
        user_query: 用户查询
        state: 父 Agent 的状态对象

    Returns:
        dict: 包含操作结果的字典
    """
    import uuid
    from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import (
        graph as file_suggestion_graph,
        get_approved_files  # 子 Agent 提供的状态访问函数
    )

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    # 可以从父状态传递上下文到子 Agent
    inputs = {
        "messages": [
            ("system", f"用户目标: {state.get('user_goal', '未指定')}"),
            ("user", user_query)
        ]
    }

    try:
        # 调用子 Agent（同步执行）
        result = file_suggestion_graph.invoke(inputs, config=config)

        # 从子 Agent 的状态中提取结果
        # 方式1: 直接读取子 Agent 的状态字段
        approved_files = result.get("approved_files", [])

        # 方式2: 调用子 Agent 提供的访问函数（如果有全局状态）
        # approved_files = get_approved_files()

        # 更新父 Agent 的状态
        state["approved_files"] = approved_files

        return {
            "status": "success",
            "approved_files": approved_files,
            "message": f"成功获取 {len(approved_files)} 个推荐文件"
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"调用失败: {str(e)}"
        }
```

## 方法三：流式调用封装

适用于需要实时获取子 Agent 执行进度的场景。

```python
@tool(description="流式调用文件建议 Agent")
@needs_state
def stream_file_suggestion_agent(
    user_query: str,
    state: ParentAgentState = None
) -> dict:
    """
    流式调用文件建议 Agent，实时获取执行进度。

    Args:
        user_query: 用户查询
        state: 父 Agent 的状态对象

    Returns:
        dict: 包含操作结果的字典
    """
    import uuid
    from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import graph

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    inputs = {"messages": [("user", user_query)]}

    try:
        messages_collected = []

        # 流式调用子 Agent
        for chunk in graph.stream(inputs, config=config, stream_mode="messages"):
            # chunk 是 (message, metadata) 元组
            msg, metadata = chunk
            messages_collected.append(msg)

            # 可以在这里处理每个消息
            # 例如：打印进度、更新UI等
            print(f"[子Agent] {msg.content if hasattr(msg, 'content') else msg}")

        # 提取最终状态
        final_state = graph.get_state(config)
        approved_files = final_state.values.get("approved_files", [])

        # 更新父状态
        state["approved_files"] = approved_files

        return {
            "status": "success",
            "approved_files": approved_files,
            "message_count": len(messages_collected)
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"流式调用失败: {str(e)}"
        }
```

## 方法四：支持人工审批的子 Agent 封装

适用于子 Agent 内部有人工审批流程的场景。

```python
@tool(description="调用需要人工审批的子 Agent")
@needs_state
def call_agent_with_approval(
    user_query: str,
    auto_approve: bool = False,
    state: ParentAgentState = None
) -> dict:
    """
    调用需要人工审批的子 Agent。

    Args:
        user_query: 用户查询
        auto_approve: 是否自动批准所有审批请求（测试用）
        state: 父 Agent 状态

    Returns:
        dict: 包含操作结果的字典
    """
    import uuid
    from utils.langgraph.OutputParser import run_workflow_with_approval_streaming
    from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import graph

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    inputs = {"messages": [("user", user_query)]}

    try:
        # 使用带审批的工作流运行器
        result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
            graph=graph,
            config=config,
            inputs=inputs,
            auto_approve=auto_approve,  # 父 Agent 可以控制是否自动批准
            debug=False
        )

        # 提取结果并更新父状态
        approved_files = result.get("approved_files", [])
        state["approved_files"] = approved_files

        return {
            "status": "success",
            "approved_files": approved_files,
            "agent_message_count": len(agent_msgs),
            "tool_message_count": len(tool_msgs)
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"调用失败: {str(e)}"
        }
```

## 完整示例：父 Agent 调用子 Agent

```python
import uuid
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import AgentState

from utils.langgraph.ChatNode import create_chat_node
from utils.langgraph.ConditionNode import should_continue
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state
from utils.models import DeepSeek_V3

# ========== 1. 定义父 Agent 的状态 ==========
class MasterAgentState(AgentState):
    user_goal: str
    approved_files: list[str]
    schema_plan: dict

# ========== 2. 封装子 Agent 为工具 ==========
@tool(description="调用文件建议 Agent 获取推荐文件")
@needs_state
def get_file_suggestions(query: str, state: MasterAgentState = None) -> dict:
    """调用文件建议子 Agent"""
    import uuid
    from Agents.KnowledgeGraphAgent.FileSuggestionAgent.FileSuggestion_Agent import graph

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {"messages": [("user", query)]}

    try:
        result = graph.invoke(inputs, config=config)
        files = result.get("approved_files", [])
        state["approved_files"] = files

        return {
            "status": "success",
            "files": files
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

@tool(description="调用 Schema 提案 Agent 生成图模式")
@needs_state
def generate_schema_plan(state: MasterAgentState = None) -> dict:
    """调用 Schema 提案子 Agent"""
    import uuid
    from Agents.KnowledgeGraphAgent.StructuredDataAgent.SchemaProposalAgent.SchemaProposalAgent import graph

    # 从父状态获取上下文
    approved_files = state.get("approved_files", [])
    user_goal = state.get("user_goal", "")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {
        "messages": [
            ("system", f"用户目标: {user_goal}"),
            ("system", f"已批准文件: {', '.join(approved_files)}"),
            ("user", "请为这些文件生成知识图谱导入方案")
        ]
    }

    try:
        result = graph.invoke(inputs, config=config)
        schema = result.get("approved_construction_plan", {})
        state["schema_plan"] = schema

        return {
            "status": "success",
            "schema": schema
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# ========== 3. 构建父 Agent ==========
tools = [get_file_suggestions, generate_schema_plan]

chat_model = DeepSeek_V3.bind_tools(tools)
tool_node = create_tool_node_with_state(tools)

master_instruction = """你是知识图谱构建的协调 Agent。
你的任务是：
1. 首先调用 get_file_suggestions 获取推荐文件
2. 然后调用 generate_schema_plan 生成图模式方案
3. 最后向用户报告完整的构建计划
"""

call_chat_node = create_chat_node(chat_model, master_instruction)

workflow = StateGraph(MasterAgentState)
workflow.add_node("agent", call_chat_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

checkpointer = InMemorySaver()
master_graph = workflow.compile(checkpointer)

# ========== 4. 运行父 Agent ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": uuid.uuid4()}}

    inputs = {
        "messages": [("user", "我想构建一个电影知识图谱")],
        "user_goal": "构建电影知识图谱"
    }

    from utils.langgraph.OutputParser import run_workflow_with_approval_streaming

    result, agent_msgs, tool_msgs = run_workflow_with_approval_streaming(
        graph=master_graph,
        config=config,
        inputs=inputs,
        debug=True
    )

    print("\n=== 最终结果 ===")
    print(f"批准的文件: {result.get('approved_files', [])}")
    print(f"Schema 方案: {result.get('schema_plan', {})}")
```

## 最佳实践

### 1. 状态隔离 vs 状态共享

- **隔离**：每个子 Agent 使用独立的 `thread_id`，适合并行调用多个子 Agent
- **共享**：通过 `@needs_state` 传递父状态，适合需要上下文的子 Agent

### 2. 错误处理

```python
@tool
@needs_state
def safe_call_subagent(query: str, state = None) -> dict:
    try:
        result = subagent_graph.invoke(...)
        return {"status": "success", "result": result}
    except TimeoutError:
        return {"status": "error", "error_message": "子 Agent 超时"}
    except Exception as e:
        return {"status": "error", "error_message": f"未知错误: {str(e)}"}
```

### 3. 超时控制

```python
import asyncio

@tool
async def call_subagent_with_timeout(query: str, timeout: int = 30) -> dict:
    """带超时的异步子 Agent 调用"""
    try:
        result = await asyncio.wait_for(
            subagent_graph.ainvoke(...),
            timeout=timeout
        )
        return {"status": "success", "result": result}
    except asyncio.TimeoutError:
        return {"status": "error", "error_message": "超时"}
```

### 4. 并行调用多个子 Agent

```python
import asyncio

async def parallel_subagents(queries: list[str]):
    """并行调用多个子 Agent 实例"""
    tasks = [
        subagent_graph.ainvoke({"messages": [("user", q)]})
        for q in queries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 常见陷阱

1. **状态污染**：子 Agent 修改了不应该共享的全局状态
   - 解决：使用独立的 `thread_id` 或深拷贝状态

2. **循环调用**：Agent A 调用 Agent B，Agent B 又调用 Agent A
   - 解决：设计清晰的调用层级，避免循环依赖

3. **内存泄漏**：长时间运行的父 Agent 累积了大量子 Agent 的 checkpoint
   - 解决：使用临时的 `InMemorySaver` 或定期清理 checkpointer

4. **审批冲突**：父 Agent 和子 Agent 都有人工审批节点
   - 解决：在封装时明确审批策略（自动批准或透传审批请求）

## 总结

将 Agent 封装为工具的核心步骤：

1. **编译子 Agent** → 得到 `graph` 对象
2. **创建 `@tool` 函数** → 封装 `graph.invoke()` 调用
3. **使用 `@needs_state`** → 实现状态共享（可选）
4. **处理返回值** → 统一返回 `{"status": "...", ...}` 格式
5. **注册到父 Agent** → 添加到 `tools` 列表

这种模式支持构建复杂的多层 Agent 架构，实现任务分解和专业化。
