# Justin Agent v0.2.0 Release Notes

## 中文

### Justin Agent v0.2.0

`v0.2.0` 是一次明确的小版本迭代，不再只是界面和交互层的小修补。这一版把新一轮 runtime 重构真正接到了主执行链路上，让 Justin 从“有模块但没接线”变成“上下文、工具、搜索、技能、证据存储都能在一轮对话里实际运行”。

### 发布定位

- 发布标签：`v0.2.0`
- 仓库：`RichGua/Justin-Agent`
- 这是在 `v0.1.0` 之后的第一次能力级迭代
- 重点不只是界面优化，而是补齐 agent runtime 的上下文与工具管线

### 本版本新增

- 会话上下文压缩与摘要
- tool event、tool fact、citation、context telemetry 数据结构
- 搜索服务与基础搜索缓存
- 可控的工具注册表与执行策略
- 技能安装、匹配与激活块注入
- 扩展点注册表
- runtime 主链路对以上能力的实际接入
- 针对搜索触发工具证据与 citation 的运行时测试

### 解决的问题

- 之前 `context/search/tools/skills/extensions` 等模块已经存在，但 runtime 仍走旧路径
- 新模块没有进入 `build_runtime_bundle()` 和 `send_message()`
- 一些新能力落了存储层和类型层，却没有进入实际一轮对话

### 用户影响

- Justin 现在可以在保守策略下为单轮对话附加工具证据
- 回答时可携带 citation 和更丰富的上下文信息
- 会话较长时可以使用压缩摘要而不是无限堆叠原始消息
- 技能与工具系统从“代码存在”变成“可被 runtime 使用”

### 验证

- `python -m unittest discover -s tests -q`

### 下一步

- 把 tool events、citations、context telemetry 暴露到 CLI 和 Web UI
- 为 search/tool/skill 流程增加更多针对性测试
- 在工具选择与搜索触发规则上继续收紧和细化

## English

### Justin Agent v0.2.0

`v0.2.0` is the first capability-focused minor release after the initial `v0.1.0` launch. This release finishes the runtime integration work so the new context, tool, search, skill, and evidence pipeline is no longer just present in the codebase but actually wired into live turns.

### Release Identity

- Release tag: `v0.2.0`
- Repository: `RichGua/Justin-Agent`
- First post-`v0.1.0` feature-level iteration
- Focused on runtime integration, not just UX polish

### Added in this release

- session context compression and summaries
- tool event, tool fact, citation, and context telemetry models
- search service with basic result caching
- controlled tool registry and execution policy
- skill install, matching, and activation block support
- extension point registry
- runtime wiring for the new pipeline in `build_runtime_bundle()` and `send_message()`
- runtime coverage for search-triggered tool evidence and citations

### Fixed

- the new `context/search/tools/skills/extensions` modules existed but were not connected to the main runtime path
- `build_runtime_bundle()` and `send_message()` were still effectively on the old flow
- newer storage/type additions were not reachable from a real turn

### User Impact

- Justin can now attach conservative tool evidence to a turn
- responses can carry citations and richer context state
- longer sessions can rely on compressed summaries instead of only raw message history
- the skill and tool system is now part of actual runtime behavior

### Validation

- `python -m unittest discover -s tests -q`

### Next Directions

- surface tool events, citations, and telemetry in the CLI and Web UI
- expand targeted test coverage for search/tool/skill flows
- tighten and refine the tool-triggering heuristics
