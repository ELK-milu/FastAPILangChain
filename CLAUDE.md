# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a FastAPI + LangChain AI Agent development project focused on creating intelligent agents using DeepSeek models via SiliconFlow API. The codebase demonstrates various agent patterns including ReAct agents, multi-agent supervisors, and knowledge graph agents.

## Common Commands

### Running the Application
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Testing
- Use the `test_main.http` file for API endpoint testing
- Available endpoints: `GET /` and `GET /hello/{name}`

## Architecture Overview

### Core Technologies
- **FastAPI** - Web framework for REST API endpoints
- **LangChain** - AI agent framework and orchestration
- **LangGraph** - State graph management for complex agent workflows
- **DeepSeek-V3** - LLM via SiliconFlow API
- **Neo4j** - Graph database for knowledge graphs

### Key Directory Structure
```
├── main.py              # FastAPI application entry point
├── utils/               # Utility functions (environment management)
├── Samples/             # Educational examples and learning materials
│   ├── reactlanggraph.py           # ReAct agent with tool calling
│   ├── SupervisorAgentSamples.py   # Multi-agent coordination
│   └── LangAgent.py                # Custom agent base class
└── Agents/              # Production agent implementations
    └── UserIntentAgent.py          # Knowledge graph use case agent
```

### Agent Architecture Patterns

#### 1. ReAct Agents (`Samples/reactlanggraph.py`)
- Reasoning and Acting pattern using LangGraph StateGraph
- Tool calling with structured inputs/outputs
- Message-based state management

#### 2. Multi-Agent Supervision (`Samples/SupervisorAgentSamples.py`)
- Supervisor agent coordinates multiple specialized agents
- Dynamic task delegation based on user queries
- Separate agents for research, mathematics, and web search

#### 3. Knowledge Graph Agents (`Agents/UserIntentAgent.py`)
- Neo4j integration for graph database operations
- Use case ideation and recommendation system
- Custom tools for graph querying and analysis

### Environment Configuration
- Configuration managed via `.env` file
- Key variables: `SILICONFLOW_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- API endpoints configured for SiliconFlow (DeepSeek model provider)

### Model Integration
- Uses SiliconFlow API as OpenAI-compatible endpoint for DeepSeek models
- Standard LangChain ChatOpenAI integration with custom base_url
- Model: `deepseek-v3` for most agent implementations

### Development Notes

#### Missing Dependencies Management
This project lacks a formal `requirements.txt`. Key dependencies include:
- `fastapi`, `uvicorn`, `langchain`, `langchain-openai`, `langchain-neo4j`, `langgraph`, `pydantic`, `python-dotenv`

#### Security Considerations
- API keys are managed via environment variables (good practice)
- Some sample files contain hardcoded API keys (avoid in production)
- Neo4j credentials are environment-based

#### Code Patterns
- Chinese comments and documentation throughout codebase
- Consistent use of LangGraph StateGraph for agent workflows
- Pydantic models for structured data validation
- Environment variable loading via `utils/env.py`