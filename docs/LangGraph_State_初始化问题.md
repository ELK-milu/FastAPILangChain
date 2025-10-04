# LangGraph State 初始化问题详解

## 问题描述

当在 LangGraph 中定义自定义 State 类时，类属性的默认值**不会自动成为 state 的初始值**。

## 问题示例

```python
from langgraph.prebuilt.chat_agent_executor import AgentState

class MyState(AgentState):
    test_num: int = 10  # 这个 10 不是初始值！

# 在工具中读取
@tool
@needs_state
def my_tool(user_input: str, state: MyState) -> dict:
    value = state.get("test_num", 0)  # 返回 0，而不是 10！
    print(f"值: {value}")  # 输出：值: 0
    return {"status": "success"}
```

## 根本原因

### 1. **类属性默认值的作用**

```python
class MyState(AgentState):
    test_num: int = 10
```

这里的 `= 10` 有两个作用：

1. **类型提示**：告诉类型检查器 `test_num` 是 `int` 类型
2. **类属性默认值**：作为类的静态属性，**不是实例初始化值**

### 2. **LangGraph State 的初始化机制**

LangGraph 的 State 是基于 **TypedDict** 实现的，而不是普通的类实例。

初始化流程：
```
graph.compile() → 创建空 state {}
    ↓
inputs 参数 → 设置初始 state
    ↓
节点执行 → 读取/修改 state
```

**关键点**：LangGraph **不会读取类定义中的默认值** 来初始化 state。

### 3. **为什么 `state.get("test_num", 0)` 返回 0？**

```python
# State 初始状态（只有 messages）
state = {"messages": [...]}

# 读取不存在的 key
value = state.get("test_num", 0)  # key 不存在，返回默认值 0
```

虽然类定义中有 `test_num: int = 10`，但：
- State 实际是一个字典：`{"messages": [...]}`
- 没有 `"test_num"` 这个 key
- `get()` 方法返回提供的默认值 `0`

## 解决方案

### 方案 1：在 inputs 中显式设置初始值（推荐）

```python
# ✅ 正确：显式设置初始值
inputs = {
    "messages": [("user", "你好")],
    "test_num": 10,  # 显式设置
    "my_dict": {},
    "my_list": []
}

result = graph.invoke(inputs, config)
```

### 方案 2：使用初始化节点

```python
def initialize_state(state: MyState):
    """初始化 state 的默认值"""
    if "test_num" not in state:
        state["test_num"] = 10
    if "my_dict" not in state:
        state["my_dict"] = {}
    return state

# 添加到工作流
workflow.add_node("init", initialize_state)
workflow.set_entry_point("init")  # 作为第一个节点
workflow.add_edge("init", "agent")
```

### 方案 3：在每个节点中处理默认值

```python
@tool
@needs_state
def my_tool(user_input: str, state: MyState) -> dict:
    # 使用 .get() 方法提供默认值
    test_num = state.get("test_num", 10)  # ✅ 如果不存在，使用 10

    # 或者使用 setdefault
    test_num = state.setdefault("test_num", 10)  # ✅ 设置并返回

    return {"status": "success", "value": test_num}
```

### 方案 4：创建 State 工厂函数

```python
def create_initial_state(user_input: str) -> dict:
    """创建带有所有默认值的初始 state"""
    return {
        "messages": [("user", user_input)],
        "test_num": 10,
        "my_dict": {},
        "my_list": [],
        "config": {"debug": False}
    }

# 使用
inputs = create_initial_state("你好")
result = graph.invoke(inputs, config)
```

## 最佳实践

### 1. **总是在 inputs 中设置所有自定义字段**

```python
class MyAgentState(AgentState):
    # 这些默认值仅用于类型提示，不是初始值
    construction_plan: dict = {}
    approved_files: list = []
    step_count: int = 0

# ✅ 正确使用
inputs = {
    "messages": [("user", "任务描述")],
    "construction_plan": {},      # 显式设置
    "approved_files": [],         # 显式设置
    "step_count": 0               # 显式设置
}
```

### 2. **使用 `setdefault()` 确保字段存在**

```python
@tool
@needs_state
def my_tool(param: str, state: MyState) -> dict:
    # 确保字段存在，不存在则设置默认值
    plan = state.setdefault("construction_plan", {})
    files = state.setdefault("approved_files", [])

    # 现在可以安全地修改
    plan["new_node"] = {...}
    files.append("file.csv")

    return {"status": "success"}
```

### 3. **文档化初始化要求**

```python
class MyAgentState(AgentState):
    """
    自定义 Agent State

    重要：创建 inputs 时必须包含以下字段：
    - messages: 消息列表（必需）
    - construction_plan: 构建计划字典（初始值：{}）
    - approved_files: 已批准文件列表（初始值：[]）
    - step_count: 步骤计数（初始值：0）

    示例：
        inputs = {
            "messages": [("user", "任务")],
            "construction_plan": {},
            "approved_files": [],
            "step_count": 0
        }
    """
    construction_plan: dict = {}
    approved_files: list = []
    step_count: int = 0
```

## 对比：TypedDict vs 普通类

### TypedDict（LangGraph 使用）
```python
from typing import TypedDict

class MyState(TypedDict):
    test_num: int

# 不会自动初始化
state = MyState()  # ❌ 错误：TypedDict 不能实例化
state = {}  # ✅ 需要手动创建字典
state["test_num"] = 10  # 手动设置
```

### 普通类
```python
class MyClass:
    test_num: int = 10  # 类属性

    def __init__(self):
        self.test_num = 10  # ✅ 实例属性，会自动初始化

obj = MyClass()
print(obj.test_num)  # 输出：10
```

## 总结

| 概念 | 说明 |
|------|------|
| **类属性默认值** | 仅用于类型提示，不是 state 初始值 |
| **State 初始化** | 通过 `inputs` 参数设置，不读取类定义 |
| **推荐做法** | 在 `inputs` 中显式设置所有自定义字段 |
| **备用方案** | 使用 `setdefault()` 或初始化节点 |

## 你的代码修复

### 修改前
```python
class SchemaProposalAgentState(AgentState):
    test_num: int = 10  # 这个 10 不起作用

inputs = {"messages": [("user", "北京市朝阳区")]}  # 缺少 test_num

# 工具中
construction_plan = state.get("test_num", 0)  # 返回 0（不是 10！）
```

### 修改后
```python
class SchemaProposalAgentState(AgentState):
    test_num: int = 10  # 仅用于类型提示

inputs = {
    "messages": [("user", "北京市朝阳区")],
    "test_num": 10  # ✅ 显式设置初始值
}

# 工具中
construction_plan = state.get("test_num", 0)  # 返回 10 ✅
```
