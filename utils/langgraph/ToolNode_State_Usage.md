# create_tool_node_with_state 使用指南

## 核心机制

`create_tool_node_with_state` 允许工具函数访问和修改 LangGraph 的 state，同时避免 Pydantic 验证错误。

## 关键要点

1. **在工具函数签名中声明 `state` 参数**
2. **使用 `@needs_state` 装饰器标记工具**
3. **直接调用原始函数，绕过 LangChain 的 `invoke` 方法**

## 完整示例

```python
from langchain_core.tools import tool
from langgraph.prebuilt.chat_agent_executor import AgentState
from utils.langgraph.ToolNode import create_tool_node_with_state, needs_state

# 定义自定义 State 类
class MyAgentState(AgentState):
    my_custom_field: dict = {"default": "value"}

# 方法 1: 使用装饰器（推荐）
@tool(description="需要访问和修改 state 的工具")
@needs_state
def my_tool_with_state(user_input: str, state: MyAgentState) -> dict:
    """
    这个工具可以访问和修改 state。

    参数:
        user_input: 用户输入
        state: Agent 状态（由 create_tool_node_with_state 自动注入，不会触发 Pydantic 验证）

    返回:
        dict: 工具执行结果
    """
    # 读取 state
    current_value = state.get("my_custom_field", {})
    print(f"当前值: {current_value}")

    # 修改 state
    state["my_custom_field"] = {"new_value": user_input}

    return {
        "status": "success",
        "old_value": current_value,
        "new_value": state["my_custom_field"]
    }

# 方法 2: 定义后调用装饰器
@tool(description="另一个需要 state 的工具")
def another_tool(data: str, state: MyAgentState) -> dict:
    return {"status": "success"}

another_tool = needs_state(another_tool)

# 定义普通工具（不需要 state）
@tool(description="普通工具")
def normal_tool(data: str) -> dict:
    """普通工具，不访问 state"""
    return {"status": "success", "data": data}

# 创建工具节点
tools = [my_tool_with_state, normal_tool]
tool_node = create_tool_node_with_state(tools)
```

## 工作原理

### 1. 为什么不能在 `@tool` 中直接声明 `state` 参数？

LangChain 的 `@tool` 装饰器会：
- 将函数的所有参数转换为 Pydantic 模型字段
- 在调用 `tool.invoke()` 时进行参数验证
- 即使参数有默认值，也会被视为必填字段

这导致如果在签名中声明 `state`，LLM 必须显式提供 `state` 参数，但 LLM 无法访问 state。

### 2. `create_tool_node_with_state` 如何解决这个问题？

```python
# 伪代码说明
def create_tool_node_with_state(tools):
    def tool_node(state):
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]

            if tool.needs_state:
                # 直接调用原始函数（绕过 Pydantic 验证）
                tool_result = tool.func(**tool_call["args"], state=state)
            else:
                # 普通工具使用标准流程
                tool_result = tool.invoke(tool_call["args"])

        return {"messages": outputs}

    return tool_node
```

关键点：
- 通过 `tool.func` 访问原始函数（未经 LangChain 包装）
- 直接调用原始函数并传入 `state` 参数
- 绕过 `tool.invoke()` 的 Pydantic 验证

### 3. `needs_state` 属性的作用

```python
my_tool.needs_state = True
```

这个属性告诉 `create_tool_node_with_state`：
- 这个工具需要 state 注入
- 应该直接调用 `tool.func` 而不是 `tool.invoke`

## 常见错误

### ❌ 错误示例 1：忘记使用 `@needs_state` 装饰器

```python
@tool
def my_tool(user_input: str, state: MyAgentState) -> dict:
    return {"status": "success"}

# 忘记添加 @needs_state 装饰器
```

结果：工具会通过 `invoke` 调用，导致 Pydantic 验证错误（`state` 参数被视为必填）。

### ❌ 错误示例 2：装饰器顺序错误

```python
@needs_state  # 错误：应该放在 @tool 之后
@tool
def my_tool(user_input: str, state: MyAgentState) -> dict:
    return {"status": "success"}
```

结果：`needs_state` 无法正确识别工具。

### ❌ 错误示例 3：尝试给 tool 对象赋值属性

```python
@tool
def my_tool(user_input: str, state: MyAgentState) -> dict:
    return {"status": "success"}

my_tool.needs_state = True  # 错误：StructuredTool 不允许动态添加属性
```

结果：`ValueError: "StructuredTool" object has no field "needs_state"`

### ✅ 正确示例

```python
@tool
@needs_state  # 正确：装饰器放在 @tool 之后
def my_tool(user_input: str, state: MyAgentState) -> dict:
    return {"status": "success"}
```

## 实际使用场景

### 场景 1：累积构建计划

```python
@tool(description="提议节点构建")
@needs_state
def propose_node(label: str, properties: list[str], state: MyState) -> dict:
    # 获取当前计划
    plan = state.get("construction_plan", {})

    # 添加新节点
    plan[label] = {"properties": properties}

    # 更新 state
    state["construction_plan"] = plan

    return {"status": "success", "plan": plan}
```

### 场景 2：跨工具共享数据

```python
@tool(description="搜索文件")
@needs_state
def search_file(file_path: str, query: str, state: MyState) -> dict:
    results = perform_search(file_path, query)

    # 保存搜索结果到 state 供其他工具使用
    state["last_search_results"] = results

    return {"status": "success", "results": results}

@tool(description="分析搜索结果")
@needs_state
def analyze_results(state: MyState) -> dict:
    # 读取之前的搜索结果
    results = state.get("last_search_results", [])
    analysis = perform_analysis(results)

    return {"status": "success", "analysis": analysis}
```

## 注意事项

1. **State 修改会持久化**：在工具中修改 state，后续节点可以访问
2. **类型提示**：使用自定义 State 类型（如 `MyAgentState`）以获得更好的 IDE 支持
3. **只在需要时使用**：不需要 state 的工具不要添加 `needs_state` 属性
4. **State 不会自动传递给 LLM**：LLM 只能通过工具返回值获取信息

## 总结

| 特性 | 说明 |
|------|------|
| **声明方式** | 在函数签名中声明 `state` 参数 |
| **标记方式** | `@needs_state` 装饰器（放在 `@tool` 之后） |
| **调用方式** | 直接调用 `tool.func()` 绕过 Pydantic 验证 |
| **State 访问** | 读取：`state.get(key)`，写入：`state[key] = value` |
| **适用场景** | 需要跨工具共享数据、累积构建计划、维护上下文 |

## 实现原理

`@needs_state` 装饰器通过全局集合 `_TOOLS_NEED_STATE` 记录工具函数的 ID：

```python
# 在 ToolNode.py 中
_TOOLS_NEED_STATE = set()

def needs_state(tool_func):
    func_to_mark = getattr(tool_func, 'func', tool_func)
    _TOOLS_NEED_STATE.add(id(func_to_mark))
    return tool_func
```

`create_tool_node_with_state` 在运行时检查工具是否在集合中：

```python
original_func = tool.func
tool_needs_state = id(original_func) in _TOOLS_NEED_STATE

if tool_needs_state:
    tool_result = original_func(**tool_args, state=state)
else:
    tool_result = tool.invoke(tool_args)
```
