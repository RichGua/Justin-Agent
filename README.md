# Justin Agent V1

Justin is a local-first agent with reviewed memory, hybrid retrieval, CLI usage, and a built-in Web dashboard.

[中文介绍](#中文) | [English](#english)

## English

### Project Introduction

`Justin Agent V1` is a local-first agent inspired by the architecture ideas behind `hermes-agent`, but rebuilt as a smaller system you can fully control.

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
- 📱 **WeChat Integration**: Native integration with personal WeChat accounts using Tencent's iLink Bot API. Supports long-polling, media CDN encryption, and QR code device flow.

### Installation

`uv` is the recommended install and runtime path for this project.

#### Quick guide

- Want to work inside the repo: clone first, then use `uv sync`
- Want a direct `Justin` command without cloning: use `uv tool install git+...`
- Want to try it once without installing: use `uvx --from git+... Justin`

#### Recommended: clone first, then use `uv` in project mode

Use this when you want an isolated project environment for development or daily use:

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
uv sync
uv run Justin
```

This opens the interactive CLI directly.
On first interactive run, Justin starts a setup wizard to choose `OPENAI`, `Ollama`, `Nvidia NIM`, or `local`.

To start the local dashboard:

```bash
uv run Justin serve
```

```bash
Justin wechat
```

#### Option B: no clone required, install `Justin` with `uv tool`

Use this when you want `Justin` available as a direct shell command and do not want to clone the repo first:

```bash
uv tool install git+https://github.com/RichGua/Justin-Agent.git
Justin
```

To start the local dashboard:

```bash
Justin serve
```

#### Option C: no clone required, run once with `uvx`

Use this when you want to try the CLI without a persistent install:

```bash
uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin
```

To start the dashboard once:

```bash
uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin serve
```

#### Option D: clone first, then install `Justin` with `uv tool --editable`

Use this when you want `Justin` as a direct command, but also want to edit the local source:

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
uv tool install --editable .
Justin
```

To start the local dashboard:

```bash
Justin serve
```

#### Option E: clone first, then use `pip` + virtual environment

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Justin
```

To start the local dashboard:

```bash
Justin serve
```

#### Option F: clone first, then run directly as a module

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m justin
```

To start the local dashboard without installing the CLI command:

```bash
python -m justin serve
```

Open `http://127.0.0.1:8765`.

### Configuration

Environment variables:

- `JUSTIN_HOME`: local data directory, default `./.justin`
- `JUSTIN_MODEL_PROVIDER`: `local`, `openai`, `ollama`, or `nvidia-nim`
- `JUSTIN_MODEL_NAME`: remote model name
- `JUSTIN_API_BASE`: OpenAI-compatible API base URL
- `JUSTIN_API_KEY`: API key for the remote provider
- `JUSTIN_MODEL_TEMPERATURE`: generation temperature, default `0.3`
- `JUSTIN_MODEL_TOP_P`: generation top_p, default `0.95`
- `JUSTIN_MODEL_MAX_TOKENS`: per-request token cap, default `1024`
- `JUSTIN_MODEL_TIMEOUT_SECONDS`: HTTP timeout, default `60`
- `JUSTIN_MODEL_RETRY_MAX_TOKENS`: retry cap when response is truncated, default `8192`
- `JUSTIN_HOST`: server host, default `127.0.0.1`
- `JUSTIN_PORT`: server port, default `8765`

Example:

```bash
set JUSTIN_MODEL_PROVIDER=openai
set JUSTIN_MODEL_NAME=gpt-4.1-mini
set JUSTIN_API_BASE=https://api.openai.com/v1
set JUSTIN_API_KEY=your_key_here
set JUSTIN_MODEL_MAX_TOKENS=2048
Justin serve
```

Run setup wizard manually anytime:

```bash
Justin setup
```

### Secret Safety

- Never commit real API keys to GitHub.
- Keep secrets in local `.env` or in Justin local settings (`.justin/settings.json`).
- `.env*` and `.justin/` are git-ignored in this repository.
- Use `.env.example` as the template and keep only placeholders in docs/snippets.

Enable the built-in pre-commit secret scanner once per clone:

```bash
git config core.hooksPath .githooks
```

Manually run the scanner at any time:

```bash
python scripts/check_secrets.py --staged
```

中文补充：
- 不要把真实密钥写进仓库文件并提交到 GitHub。
- 推荐把密钥写到本地 `.env`（已被忽略）或 `.justin/settings.json`（目录已被忽略）。
- 首次克隆后执行 `git config core.hooksPath .githooks`，提交前会自动扫描密钥。

### CLI Usage

```bash
Justin
Justin chat --message "remember I prefer concise output"
Justin chat --session <session-id>
Justin candidate list
Justin candidate confirm <candidate-id>
Justin candidate reject <candidate-id> --note "not stable enough"
Justin memory list
Justin memory search "concise output"
```

Interactive slash commands:

```text
/help /session /provider /stats /theme /new /setup /candidates /approve /reject /memories /clear /exit
```

### Notes

- Memory writes are conservative by design: candidate memories require approval.
- SQLite is the only required storage dependency.
- The local fallback model is useful for testing the memory workflow, but a stronger remote model will produce better responses.
- If you use `uv sync`, prefer `uv run Justin ...` inside the project directory.
- If you do not want to clone the repo first, prefer `uv tool install git+https://github.com/RichGua/Justin-Agent.git`.
- If you only want to try it once, prefer `uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin`.
- If you want `Justin` available as a direct command everywhere, use `uv tool install --editable .`.

## 中文

### 项目介绍

`Justin Agent V1` 是一个本地优先的个人 Agent，设计思路借鉴了 `hermes-agent`，但实现上更轻量、可控，适合你自己长期迭代和掌控。

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

推荐优先使用 `uv`。

#### 快速选择

- 想在仓库里开发或长期维护：先 `clone`，再用 `uv sync`
- 想不 clone 直接得到 `Justin` 命令：用 `uv tool install git+...`
- 想不安装、先试一次：用 `uvx --from git+... Justin`

#### 推荐方式：先拉取仓库，再使用 `uv` 的项目模式

如果你希望在项目目录里使用隔离环境进行开发或日常运行：

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
uv sync
uv run Justin
```

这会直接进入交互式 CLI。
首次进入交互模式时，Justin 会启动配置向导，支持 `OPENAI`、`Ollama`、`Nvidia NIM`、`local`。

如果要启动本地 Web 面板：

```bash
uv run Justin serve
```

#### 方式二：无需 clone，直接用 `uv tool` 安装命令

如果你希望不先拉取仓库、安装后就能直接输入 `Justin`：

```bash
uv tool install git+https://github.com/RichGua/Justin-Agent.git
Justin
```

启动本地 Web 面板：

```bash
Justin serve
```

#### 方式三：无需 clone，使用 `uvx` 直接试运行

如果你只是想先体验一下，不想做持久安装：

```bash
uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin
```

一次性启动本地 Web 面板：

```bash
uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin serve
```

#### 方式四：先拉取仓库，再用 `uv tool --editable` 安装

如果你希望既能直接输入 `Justin`，又要保留本地源码可编辑能力：

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
uv tool install --editable .
Justin
```

启动本地 Web 面板：

```bash
Justin serve
```

#### 方式五：先拉取仓库，再使用 `pip` + 虚拟环境

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Justin
```

如果要启动本地 Web 面板：

```bash
Justin serve
```

#### 方式六：先拉取仓库，再直接以模块方式运行

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m justin
```

如果你暂时不想安装命令入口，可以直接这样运行。

启动本地 Web 面板：

```bash
python -m justin serve
```

然后打开 `http://127.0.0.1:8765`。

### 配置说明

可用环境变量：

- `JUSTIN_HOME`：本地数据目录，默认 `./.justin`
- `JUSTIN_MODEL_PROVIDER`：`local`、`openai`、`ollama` 或 `nvidia-nim`
- `JUSTIN_MODEL_NAME`：远程模型名
- `JUSTIN_API_BASE`：兼容 OpenAI 的接口地址
- `JUSTIN_API_KEY`：远程模型 API Key
- `JUSTIN_HOST`：服务监听地址，默认 `127.0.0.1`
- `JUSTIN_PORT`：服务端口，默认 `8765`

示例：

```bash
set JUSTIN_MODEL_PROVIDER=openai
set JUSTIN_MODEL_NAME=gpt-4.1-mini
set JUSTIN_API_BASE=https://api.openai.com/v1
set JUSTIN_API_KEY=your_key_here
Justin serve
```

你也可以随时手动重跑配置向导：

```bash
Justin setup
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
- 如果你使用 `uv sync`，建议在项目目录里用 `uv run Justin ...`。
- 如果你不想先 clone 仓库，建议优先使用 `uv tool install git+https://github.com/RichGua/Justin-Agent.git`。
- 如果你只是想先试一次，建议用 `uvx --from git+https://github.com/RichGua/Justin-Agent.git Justin`。
- 如果你希望系统里可以直接输入 `Justin`，建议使用 `uv tool install --editable .`。
