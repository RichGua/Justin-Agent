# v0.1.0 Release Notes

## 中文

### Justin-Agent v0.1.0 首发版本

这是 `Justin-Agent` 的第一个可用版本。当前版本聚焦在“本地优先 + 可审核长期记忆 + 可追溯回答”这条主链路，目标不是一开始就做成全自动 Agent，而是先把个人长期使用最关键的记忆闭环做稳。

### 本版本包含

- 本地优先的个人 Agent 基础架构
- SQLite 持久化会话与长期记忆存储
- 候选记忆审核机制：先提议，再确认入库
- 已确认长期记忆的混合检索
- CLI 交互入口
- 本地 Web 仪表盘
- 可替换模型适配层
- 基础测试用例与运行说明

### 适合谁

这个版本适合想先搭建“属于自己的 Agent 系统”并从个人偏好、长期目标、项目上下文逐步积累记忆的开发者或个人用户。

### 当前特点

- 默认保守：长期记忆不会自动写入，必须确认
- 本地可运行：不强依赖云端数据库或第三方后台
- 可追溯：回答可以关联到被召回的已确认记忆
- 易扩展：后续可以继续接更强模型、任务系统、定时器、插件和多入口

### 已验证

- 单元测试通过
- CLI 聊天链路可用
- HTTP API 可用
- 本地 Web 面板可启动并使用

### 下一步方向

- 更强的模型接入与更好的个性化回答质量
- 更细粒度的记忆分类与过期/修正机制
- 任务管理、提醒与长期目标跟进
- 插件化工具系统与更强的执行能力

## English

### Justin-Agent v0.1.0 Initial Release

This is the first usable release of `Justin-Agent`. The current version focuses on the core loop of `local-first usage + reviewable long-term memory + traceable responses`. It is intentionally not trying to be a fully autonomous agent on day one; the goal is to build a solid personal memory system first.

### Included in this release

- Local-first Justin foundation
- SQLite-backed session and long-term memory persistence
- Candidate-memory review workflow
- Hybrid retrieval over approved memories
- CLI interface
- Built-in local Web dashboard
- Pluggable model adapter layer
- Basic tests and setup documentation

### Who this is for

This release is meant for developers and personal users who want to build their own agent system and gradually accumulate stable memory from preferences, long-term goals, and project context.

### Current characteristics

- Conservative by default: long-term memory requires explicit approval
- Runs locally: no hard dependency on a cloud database or hosted backend
- Traceable: responses can be tied back to recalled approved memories
- Extendable: ready for stronger models, task systems, schedulers, plugins, and more interfaces later

### Verified in this version

- Unit tests passed
- CLI chat flow works
- HTTP API works
- Local Web dashboard starts and works

### Next directions

- Stronger model integrations and better personalization quality
- Finer-grained memory types plus correction and expiration workflows
- Task tracking, reminders, and long-horizon goal follow-up
- A plugin-style tool system and stronger execution capabilities
