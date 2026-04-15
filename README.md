# Personal Agent V1

Local-first personal agent with reviewed memory, hybrid retrieval, CLI usage, and a built-in Web dashboard.

[中文介绍](#中文) | [English](#english)

## English

### Project Introduction

`Personal Agent V1` is a local-first personal agent inspired by the architecture ideas behind `hermes-agent`, but rebuilt as a smaller system you can fully control.

This project is designed for a single-user workflow:

- it keeps persistent chat sessions
- it extracts candidate memories from conversations
- it requires explicit approval before long-term memory is stored
- it retrieves approved memories before every response
- it provides both a CLI and a local Web UI
- it supports a pluggable model layer, with a dependency-free local fallback

The current version focuses on a safe memory loop rather than full autonomy. It is meant to become more useful over time by remembering your stable preferences, goals, identity details, and project context.

### Core Capabilities

- `Persistent sessions`: all conversations are stored in SQLite
- `Candidate memory review`: the agent proposes memories instead of writing them directly
- `Approved long-term memory`: only confirmed memories are used as trusted background
- `Hybrid retrieval`: keyword-style matching plus lightweight semantic similarity
- `Traceability`: each response can show which approved memories were recalled
- `Local dashboard`: chat, candidate review, and memory browsing in one place

### Installation

#### Option A: run directly from source

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m personal_agent serve
```

Open `http://127.0.0.1:8765`.

#### Option B: use the installed CLI entrypoint

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Justin
```

This opens the interactive CLI directly. To start the local dashboard:

```bash
Justin serve
```

### Configuration

Environment variables:

- `PERSONAL_AGENT_HOME`: local data directory, default `./.personal_agent`
- `PERSONAL_AGENT_MODEL_PROVIDER`: `local` or `openai-compatible`
- `PERSONAL_AGENT_MODEL_NAME`: remote model name
- `PERSONAL_AGENT_API_BASE`: OpenAI-compatible API base URL
- `PERSONAL_AGENT_API_KEY`: API key for the remote provider
- `PERSONAL_AGENT_HOST`: server host, default `127.0.0.1`
- `PERSONAL_AGENT_PORT`: server port, default `8765`

Example:

```bash
set PERSONAL_AGENT_MODEL_PROVIDER=openai-compatible
set PERSONAL_AGENT_MODEL_NAME=gpt-4.1-mini
set PERSONAL_AGENT_API_BASE=https://api.openai.com/v1
set PERSONAL_AGENT_API_KEY=your_key_here
personal-agent serve
```

### CLI Usage

```bash
Justin
Justin chat --message "remember I prefer concise output"
Justin candidate list
Justin candidate confirm <candidate-id>
Justin candidate reject <candidate-id> --note "not stable enough"
Justin memory list
Justin memory search "concise output"
```

### Notes

- Memory writes are conservative by design: candidate memories require approval.
- SQLite is the only required storage dependency.
- The local fallback model is useful for testing the memory workflow, but a stronger remote model will produce better responses.

## 中文

### 项目介绍

`Personal Agent V1` 是一个本地优先的个人 Agent，设计思路借鉴了 `hermes-agent`，但实现上更轻量、可控，适合你自己长期迭代和掌控。

这个项目面向单用户长期使用场景，核心目标不是“全自动万能助手”，而是一个会逐步了解你、但在长期记忆写入上保持保守和可审核的个人系统：

- 保存持续会话
- 从对话中提炼候选记忆
- 必须经人工确认后才写入长期记忆
- 每次回答前先检索已确认记忆
- 同时提供 CLI 和本地 Web 面板
- 支持可替换的模型接入层，并内置无第三方依赖的本地回退模型

当前版本重点解决的是“记忆闭环”而不是完全自治。它会随着使用逐步积累你的稳定偏好、身份背景、长期目标和项目上下文，从而让后续回答越来越贴近你本人。

### 核心能力

- `持久化会话`：所有对话都存入 SQLite
- `候选记忆审核`：Agent 先提出记忆候选，不直接写库
- `已确认长期记忆`：只有确认后的记忆才会被当作可靠背景
- `混合检索`：关键词召回 + 轻量语义相似度
- `可追溯性`：每次回答都可以看到命中了哪些长期记忆
- `本地仪表盘`：聊天、审核候选记忆、浏览记忆都在一个界面内完成

### 安装说明

#### 方式一：直接从源码运行

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m personal_agent serve
```

然后打开 `http://127.0.0.1:8765`。

#### 方式二：使用安装后的 CLI 命令

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Justin
```

这会直接进入交互式 CLI。如果要启动本地 Web 面板：

```bash
Justin serve
```

### 配置说明

可用环境变量：

- `PERSONAL_AGENT_HOME`：本地数据目录，默认 `./.personal_agent`
- `PERSONAL_AGENT_MODEL_PROVIDER`：`local` 或 `openai-compatible`
- `PERSONAL_AGENT_MODEL_NAME`：远程模型名
- `PERSONAL_AGENT_API_BASE`：兼容 OpenAI 的接口地址
- `PERSONAL_AGENT_API_KEY`：远程模型 API Key
- `PERSONAL_AGENT_HOST`：服务监听地址，默认 `127.0.0.1`
- `PERSONAL_AGENT_PORT`：服务端口，默认 `8765`

示例：

```bash
set PERSONAL_AGENT_MODEL_PROVIDER=openai-compatible
set PERSONAL_AGENT_MODEL_NAME=gpt-4.1-mini
set PERSONAL_AGENT_API_BASE=https://api.openai.com/v1
set PERSONAL_AGENT_API_KEY=your_key_here
personal-agent serve
```

### CLI 示例

```bash
Justin
Justin chat --message "记住我偏好简洁输出"
Justin candidate list
Justin candidate confirm <candidate-id>
Justin candidate reject <candidate-id> --note "这条还不够稳定"
Justin memory list
Justin memory search "简洁 输出"
```

### 说明

- 长期记忆写入默认是保守的：候选记忆必须先审核再入库。
- 存储层只强依赖 SQLite。
- 本地回退模型适合验证记忆流程，真正想要更强的对话质量，建议接入更强的远程模型。
