# Justin Agent

Justin 是一款**轻量级、本地优先**的个人 AI Agent 系统。它摒弃了过度复杂的云端沙盒和沉重的环境依赖，专注于提供一套极其稳定、拥有智能长期记忆、并且能够直接接入微信的智能助手核心。

Justin 的设计灵感来源于优秀的 Hermes-Agent 架构，但 Justin 选择了一条更加**精简且完全受你掌控**的道路：它不需要你安装 Docker，不需要配置云端沙盒，只需要 Python 和 SQLite 即可在任何环境（无论是个人电脑还是低配 VPS）顺畅运行。

[English Documentation](#english-documentation)

---

## 🌟 核心特性与功能

### 1. 🧠 具备审核机制的“长期智能记忆”
Justin 不是一个聊完即忘的聊天机器人，也不是一个随便胡乱记录垃圾信息的系统。
- **记忆候选与审核（Candidate Review）**：Justin 会在对话中自动分析并提取关于你的偏好、习惯、项目背景和长期目标，形成“记忆候选（Candidate）”。这些记忆必须经过你的**明确批准（Approve）**才会入库。
- **混合检索（Hybrid Retrieval）**：每次对话前，Justin 都会根据上下文，通过内置的“关键词 + 语义相似度”混合检索机制，精准召回那些你已经批准过的相关长期记忆。
- **上下文智能压缩（Auto Compression）**：当当前会话（Session）的 Token 接近阈值时，Justin 会触发自我评估机制（`evaluate_compressibility`），并在后台悄悄地将长篇对话总结为精炼的摘要，永远保持上下文窗口的清爽。

### 2. 📱 深度微信集成 (WeChat iLink API)
通过腾讯官方的 iLink Bot API 协议，Justin 可以将自己化身为你的微信私人助理：
- **网关式接入流程**：先运行 `justin gateway setup` 完成账号配对和接入策略配置，再用 `justin gateway start` 或兼容入口 `justin wechat` 启动消息网关。
- **原生扫码登录与持久化会话**：首次配对会打印巨大且纯净的 ASCII 二维码，扫码成功后会把账号凭证保存在本地，后续启动无需每次重新扫码。
- **防封号长轮询**：底层采用工业级的 `aiohttp` 异步网络库，支持长轮询（Long-polling）机制，杜绝频繁请求导致的封号或超时断连风险。
- **自动回复防混淆**：你可以为 Justin 配置专门的微信回复前缀（如 `[Justin] `），让朋友一眼看出这是你的 AI 助理在代为回复。
- **自定义 App ID 与访问控制**：支持在网关向导中填入真实的 `wechat_app_id`，并配置 `open / allowlist / disabled` 接入策略、管理员账号和允许名单。

### 3. ⚙️ 强大的终端交互与拦截机制
在终端 CLI 环境下，Justin 拥有极佳的用户体验：
- **富文本终端 UI**：基于 `rich` 库构建的漂亮气泡、颜色区分和 Markdown 渲染。
- **工具执行安全拦截**：当 Agent 试图执行某些危险的本地工具时，基于 `prompt_toolkit` 构建的拦截界面会弹出，允许你通过键盘方向键选择 `[Yes / No / Allow for this turn]`。
- **幕后深度思考（Hidden Reasoning）**：针对 `deepseek-reasoner` 等深度推理模型，Justin 能够在后台流式接收并处理长达几千字的思想链（Reasoning Content），但在最终的 CLI 界面中将其巧妙隐藏，只展示最终回复，保持屏幕清爽。
- **智能截断重试**：当模型因为 `max_tokens` 限制导致回答截断时（`finish_reason="length"`），底层的 `openai` 架构会自动翻倍 token 阈值并静默重试，确保你每次都能看到完整的回答。

### 4. 🤖 全自动巡航模式 (Auto Mode)
使用 `/auto` 指令，Justin 将进入“自动巡航”模式。
- 支持最高 100 次的工具连续迭代调用（Max 100 tool iterations）。
- 它能自己查资料、自己写代码、自己运行验证，直到任务完成。
- **任务结束后反思（Self-reflection）**：当 Auto 模式完成一项大任务后，Justin 会自动触发内部反思流程，总结在这个任务中学到的新知识，并尝试提取出对未来有用的长期记忆。

### 5. 🔌 极简且稳健的模型提供商配置 (Setup Wizard)
只需执行一次 `justin setup`，友好的向导将带你完成所有的核心配置：
- **Agent 身份设定**：为你的 Agent 命名（默认为 Justin），并自定义它的 System Prompt 核心设定。
- **广泛的模型支持**：原生支持 OpenAI、Ollama（本地模型）、DeepSeek、Groq、OpenRouter、Together AI、Nvidia NIM，以及任何兼容 OpenAI 格式的 API 接口。
- **与网关解耦的配置流程**：核心模型配置仍由 `justin setup` 处理；微信平台接入则交给独立的 `justin gateway setup`，更接近 Hermes 的工程化流程。

---

## 🚀 安装与启动

推荐使用现代的 `uv` 包管理器进行极速安装和环境隔离。

### 方式 1：推荐安装方式 (Clone & uv)
适合希望在本地保留源码、方便随时魔改和升级的用户。

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent

# 使用 uv 同步依赖
uv sync

# 进入交互式聊天
uv run justin
```

### 方式 2：全局工具安装
适合把 Justin 当作一个全局命令随时随地调用的用户。

```bash
uv tool install git+https://github.com/RichGua/Justin-Agent.git

# 随时随地唤起
justin
```

### 方式 3：传统 Pip 安装
如果你没有安装 uv，也可以使用标准的 Python 环境：

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m venv .venv
source .venv/bin/activate  # Windows 下为 .venv\Scripts\activate
pip install -e .

justin
```

---

## 🎮 使用指南

### 1. 初始配置向导
首次运行，或者随时想更改配置，请执行：
```bash
justin setup
```
向导将引导你配置大模型 API 密钥、设定 Agent 名字，以及是否开启 WeChat gateway。

### 2. 交互式 CLI 对话
```bash
justin
```
在 CLI 中，你可以随时输入 `/help` 查看所有可用的快捷指令：
- `/auto`：开启/关闭自动工具调用模式。
- `/compact`：立刻使用 LLM 总结并压缩当前会话的上下文。
- `/context_limit <num>`：动态修改允许输入给大模型的最大 Token 阈值。
- `/tokens <num>`：动态修改模型输出的最大 Token 限制。
- `/gateway`：查看当前 WeChat gateway 的启用状态、策略和已配对账号。
- `/memories`：查看你目前已经批准的长期记忆。
- `/candidates`：查看等待你审核的候选记忆。

### 3. 启动微信助理
建议使用两段式流程：
```bash
justin gateway setup
justin gateway start
```
`justin gateway setup` 会完成 WeChat iLink 账号配对、自动回复前缀和访问策略设置，并把配对凭证保存到本地。之后运行 `justin gateway start` 就能直接恢复网关；如果你更喜欢老入口，`justin wechat` 仍然可用，并会在缺少配对信息时自动引导扫码。

---

## 🗄️ 数据存储与隐私
- **纯本地存储**：你所有的对话历史、长期记忆数据库、配置文件全部保存在 `~/.justin`（或你指定的 `JUSTIN_HOME`）目录下的 SQLite 数据库和本地 JSON 文件中。
- **0 遥测**：Justin 不包含任何隐蔽的遥测（Telemetry）或数据上报逻辑，你的隐私完全属于你自己。

---

# English Documentation

# Justin Agent

Justin is a **lightweight, local-first** personal AI Agent system. It strips away overly complex cloud sandboxes and heavy environment dependencies, focusing instead on providing an extremely stable core intelligent assistant equipped with smart long-term memory and direct WeChat integration capabilities.

Inspired by the excellent Hermes-Agent architecture, Justin chooses a much more **streamlined and fully user-controlled** path: it requires no Docker, no cloud sandbox setups, and only needs Python and SQLite to run smoothly in any environment (from personal laptops to low-end VPS).

---

## 🌟 Core Features & Capabilities

### 1. 🧠 "Long-Term Memory" with a Review Mechanism
Justin is not a chat bot that forgets everything after a session, nor is it a system that blindly records garbage information.
- **Candidate Review**: During conversations, Justin automatically analyzes and extracts your preferences, habits, project backgrounds, and long-term goals to form "Memory Candidates." These memories must be **explicitly approved** by you before being stored in the database.
- **Hybrid Retrieval**: Before each conversation, Justin uses a built-in "keyword + semantic similarity" hybrid retrieval mechanism to precisely recall the relevant long-term memories you have already approved based on the context.
- **Auto Context Compression**: When the current session's Token count approaches the threshold, Justin triggers a self-evaluation mechanism (`evaluate_compressibility`) and quietly summarizes long conversations into concise summaries in the background, keeping the context window forever clean.

### 2. 📱 Deep WeChat Integration (WeChat iLink API)
Through Tencent's official iLink Bot API protocol, Justin can transform into your personal WeChat assistant:
- **Gateway-style workflow**: Run `justin gateway setup` to pair the account and configure access policy, then use `justin gateway start` or the compatibility entrypoint `justin wechat` to bring the gateway online.
- **Native QR login with persisted sessions**: The first pairing prints a large ASCII QR code, and the resulting account credentials are saved locally so subsequent starts can restore the gateway without forcing a fresh scan.
- **Anti-Ban Long-Polling**: The underlying architecture uses the industrial-grade `aiohttp` async network library, supporting long-polling mechanisms to eliminate the risk of bans or timeouts caused by frequent requests.
- **Anti-Confusion Auto-Reply**: You can configure a specific WeChat reply prefix for Justin (e.g., `[Justin] `) so your friends can instantly recognize that your AI assistant is replying on your behalf.
- **Custom App ID and access control**: The gateway setup flow supports a real `wechat_app_id`, plus `open / allowlist / disabled` access policy, an admin user, and an explicit allowlist.

### 3. ⚙️ Powerful Terminal Interaction & Interception Mechanisms
In the CLI environment, Justin offers an excellent user experience:
- **Rich Terminal UI**: Beautiful speech bubbles, color differentiation, and Markdown rendering built on the `rich` library.
- **Tool Execution Security Interception**: When the Agent attempts to execute potentially dangerous local tools, an interception interface built on `prompt_toolkit` pops up, allowing you to use keyboard arrow keys to select `[Yes / No / Allow for this turn]`.
- **Hidden Reasoning**: For deep reasoning models like `deepseek-reasoner`, Justin can stream and process chains of thought (Reasoning Content) up to thousands of words in the background, cleverly hiding them from the final CLI interface to display only the final reply, keeping the screen clean.
- **Smart Truncation Retry**: When the model's answer is truncated due to `max_tokens` limits (`finish_reason="length"`), the underlying `openai` architecture automatically doubles the token threshold and silently retries, ensuring you always see the complete answer.

### 4. 🤖 Fully Automatic Cruise Mode (Auto Mode)
Using the `/auto` command, Justin enters "Auto Cruise" mode.
- Supports up to 100 consecutive tool iteration calls.
- It can look up information, write code, and run verifications on its own until the task is complete.
- **Self-reflection After Tasks**: When Auto mode finishes a major task, Justin automatically triggers an internal reflection process, summarizing the new knowledge learned during the task and attempting to extract useful long-term memories for the future.

### 5. 🔌 Minimalist and Robust Model Provider Configuration (Setup Wizard)
By running `justin setup` just once, a friendly wizard will guide you through all core configurations:
- **Agent Identity Setup**: Name your Agent (default is Justin) and customize its core System Prompt settings.
- **Broad Model Support**: Natively supports OpenAI, Ollama (local models), DeepSeek, Groq, OpenRouter, Together AI, Nvidia NIM, and any API interface compatible with the OpenAI format.
- **Decoupled gateway onboarding**: Core model setup stays in `justin setup`, while messaging-platform pairing lives in `justin gateway setup`, which is closer to the Hermes-style setup flow.

---

## 🚀 Installation & Startup

It is highly recommended to use the modern `uv` package manager for fast installation and environment isolation.

### Method 1: Recommended (Clone & uv)
Suitable for users who want to keep the source code locally for easy modification and upgrading.

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent

# Use uv to sync dependencies
uv sync

# Enter interactive chat
uv run justin
```

### Method 2: Global Tool Installation
Suitable for users who want to call Justin as a global command anytime, anywhere.

```bash
uv tool install git+https://github.com/RichGua/Justin-Agent.git

# Call anytime
justin
```

### Method 3: Traditional Pip Installation
If you don't have uv installed, you can also use a standard Python environment:

```bash
git clone https://github.com/RichGua/Justin-Agent.git
cd Justin-Agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

justin
```

---

## 🎮 User Guide

### 1. Initial Setup Wizard
For the first run, or whenever you want to change configurations, execute:
```bash
justin setup
```
The wizard will guide you through the LLM provider, API key, agent identity, and whether to enable the WeChat gateway.

### 2. Interactive CLI Chat
```bash
justin
```
In the CLI, you can type `/help` at any time to see all available shortcut commands:
- `/auto`: Enable/disable automatic tool call mode.
- `/compact`: Immediately use the LLM to summarize and compress the context of the current session.
- `/context_limit <num>`: Dynamically modify the maximum input Token threshold allowed for the LLM.
- `/tokens <num>`: Dynamically modify the maximum output Token limit of the model.
- `/gateway`: Show whether the WeChat gateway is enabled, which policy it uses, and which account is paired.
- `/memories`: View the long-term memories you have currently approved.
- `/candidates`: View candidate memories waiting for your review.

### 3. Launch WeChat Assistant
The recommended flow is:
```bash
justin gateway setup
justin gateway start
```
`justin gateway setup` handles iLink pairing, reply-prefix settings, and gateway access control, then saves the paired account locally. After that, `justin gateway start` restores the saved credentials and starts listening immediately. The legacy shortcut `justin wechat` still works and will fall back to QR pairing when needed.

---

## 🗄️ Data Storage & Privacy
- **Pure Local Storage**: All your chat histories, long-term memory databases, and configuration files are saved locally in SQLite databases and JSON files under the `~/.justin` (or your specified `JUSTIN_HOME`) directory.
- **Zero Telemetry**: Justin contains no hidden telemetry or data reporting logic; your privacy is entirely yours.
