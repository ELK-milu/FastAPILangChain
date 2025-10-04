# State 初始化问题诊断与解决

## 问题现象

```python
class SchemaProposalAgentState(AgentState):
    test_num: int = 10

inputs = {
    "messages": [("user", "北京市朝阳区")],
    "test_num": 10
}

# 在工具中
state.get("test_num", 0)  # 返回 0 或 0.0，而不是 10！
state["test_num"]  # KeyError 或返回 0.0
```

## 根本原因

### 原因 1：AgentState 的 Reducer 配置

`AgentState` 使用了特殊的 Reducer 来合并 state：

```python
# LangGraph 内部实现（简化版）
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 有 Reducer
    remaining_steps: int  # 内部字段
    # 自定义字段没有 Reducer！
```

**问题**：
- `messages` 字段有 `add_messages` Reducer，会正确合并
- 自定义字段（如 `test_num`）没有 Reducer
- 节点返回值会**覆盖**而不是合并这些字段

### 原因 2：节点返回值的合并逻辑

```python
# ChatNode 返回
return {"messages": [response]}  # 只返回 messages

# LangGraph 合并逻辑（伪代码）
new_state = {}
for key in node_return:
    if has_reducer(key):
        new_state[key] = reducer(old_state[key], node_return[key])
    else:
        new_state[key] = node_return[key]  # 直接覆盖

for key in old_state:
    if key not in node_return:
        if has_default():
            new_state[key] = default_value()  # ← test_num 被重置为 0！
        else:
            new_state[key] = old_state[key]
```

## 解决方案

### 方案 1：为自定义字段添加 Reducer（推荐）

```python
from typing import Annotated
from operator import add

def preserve_value(old, new):
    """保留值的 Reducer：如果新值不存在，使用旧值"""
    return new if new is not None else old

class SchemaProposalAgentState(AgentState):
    # 使用 Annotated 添加 Reducer
    test_num: Annotated[int, preserve_value]

    # 或者使用 lambda
    construction_plan: Annotated[dict, lambda old, new: new if new else old]
```

### 方案 2：节点返回所有字段

```python
def call_model(state: AgentState, config: RunnableConfig):
    response = model.invoke([prompt] + state["messages"], config)

    # 返回所有字段，包括未修改的
    return {
        "messages": [response],
        "test_num": state.get("test_num", 10),  # 保留或使用默认值
        "construction_plan": state.get("construction_plan", {})
    }
```

### 方案 3：使用简单的 TypedDict（不继承 AgentState）

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class SimpleState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    test_num: int
    construction_plan: dict

# 使用时必须显式设置所有字段
inputs = {
    "messages": [("user", "任务")],
    "test_num": 10,
    "construction_plan": {}
}
```

### 方案 4：自定义 Reducer 函数

```python
from typing import Annotated

def merge_preserving(old, new):
    """
    合并 Reducer：新值优先，但保留旧值中的字段
    适用于 dict 类型
    """
    if old is None:
        return new
    if new is None:
        return old
    return {**old, **new}

class SchemaProposalAgentState(AgentState):
    test_num: int  # 简单类型，使用默认覆盖
    construction_plan: Annotated[dict, merge_preserving]  # 字典类型，使用合并
```

## 最佳实践（推荐方案）

### 1. 使用明确的 Reducer 配置

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# 定义 Reducer
def keep_latest(old, new):
    """保留最新值"""
    return new if new is not None else old

def merge_dict(old, new):
    """合并字典"""
    if not old:
        return new or {}
    if not new:
        return old
    return {**old, **new}

# 定义 State
class SchemaProposalAgentState(TypedDict):
    # messages 使用 add_messages Reducer（追加消息）
    messages: Annotated[list[BaseMessage], add_messages]

    # test_num 使用 keep_latest Reducer（保留最新值）
    test_num: Annotated[int, keep_latest]

    # construction_plan 使用 merge_dict Reducer（合并字典）
    construction_plan: Annotated[dict, merge_dict]
```

### 2. 初始化所有字段

```python
inputs = {
    "messages": [("user", "任务描述")],
    "test_num": 10,
    "construction_plan": {}
}
```

### 3. 节点只返回需要更新的字段

```python
def chat_node(state):
    response = model.invoke(state["messages"])
    # 只返回 messages，其他字段由 Reducer 保留
    return {"messages": [response]}

def tool_node(state):
    result = execute_tool(state)
    # 更新 construction_plan，其他字段保留
    return {
        "construction_plan": {
            "new_node": result["node_data"]
        }
    }
```

## 调试技巧

### 1. 打印 State 的完整内容

```python
def my_node(state):
    print(f"State keys: {list(state.keys())}")
    print(f"State content: {dict(state)}")
    print(f"test_num exists: {'test_num' in state}")
    print(f"test_num value: {state.get('test_num', 'NOT_FOUND')}")
    print(f"test_num type: {type(state.get('test_num'))}")
```

### 2. 检查节点返回值

```python
def my_node(state):
    result = {"messages": [response]}
    print(f"Node returning: {result}")
    return result
```

### 3. 使用 Graph 的 debug 模式

```python
result = graph.invoke(inputs, config, debug=True)
```

## 你的代码修复方案

### 当前问题

```python
class SchemaProposalAgentState(AgentState):
    test_num: int = 10  # 没有 Reducer，会被重置

# ChatNode 只返回 messages
return {"messages": [response]}  # test_num 被重置为 0 或丢失
```

### 推荐修复

```python
from typing import Annotated

def keep_value(old, new):
    """如果新值为 None，保留旧值"""
    return new if new is not None else old

class SchemaProposalAgentState(AgentState):
    test_num: Annotated[int, keep_value]  # 添加 Reducer

# 初始化
inputs = {
    "messages": [("user", "北京市朝阳区")],
    "test_num": 10
}

# ChatNode 保持不变（只返回 messages）
def call_model(state, config):
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # test_num 由 Reducer 自动保留
```

## 总结

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| State 字段被重置 | 没有 Reducer 配置 | 添加 Reducer |
| 字段值变为 0.0 | 默认值类型推断错误 | 显式设置初始值 |
| 无法读取字段 | 字段不存在 | 在 inputs 中初始化 |
| 字典被覆盖 | 使用了覆盖而非合并 | 使用 merge Reducer |
