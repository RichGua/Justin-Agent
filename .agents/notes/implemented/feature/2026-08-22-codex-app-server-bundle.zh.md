# Agent Note: 将 Codex App Server 作为可选 Harness Bundle 交付

Status: implemented

[English](2026-08-22-codex-app-server-bundle.md) | 中文

## Problem

Justin 需要保留 DeepSeek Harness 可替换的 Cordis 产品结构，同时把官方 Codex App Server 作为可选的执行产品。机器上的 `codex` 可执行文件不是可靠的兼容性边界，因为其版本和协议可独立于 Harness 漂移。

## Decision

`@deepseek-ai/dsh-subagent-codex` 是仓库内的 Codex 产品 Bundle。它将 `@openai/codex` 固定为 `0.149.0`，解析该包自身声明的可执行文件，并且只通过 `app-server --stdio` 使用 JSON-RPC。Provider 挂载后仍处于休眠状态：只有已启用的 Agent Preset 委派工具调用时才启动进程。

Web Profile 模板包含此 Bundle，因此新的本地 Profile 无需额外从 registry 安装即可具备 Provider。标准 Preset 仍将 `subagent_codex` 保持禁用。用户通过复制 Preset 并启用该工具行来打开它；从 Profile 中移除 Bundle 后，Provider 会在下次重启时撤销。这使产品可用性和 Agent 权限保持独立。

## Alternatives considered

**再写一套 Codex 代理循环** — 拒绝。App Server 负责线程、Turn、审批、沙箱、账户、模型、MCP、技能和原生插件行为；复制该循环必然产生偏差。

**使用 `PATH` 中找到的 `codex` 可执行文件** — 拒绝。它会让不同机器静默使用不同协议基线。包内的 `0.149.0` wrapper 会选择对应的原生载荷。

**在每个新会话中启用委派工具** — 拒绝。挂载 Provider 不应自动授予模型工具或在用户未选择时启动任务。

## Consequences

新的 Web Profile 包含 Codex 产品能力，但默认不会创建 App Server 进程，也不会将其暴露给 Agent。浏览器表层保留全部原生 DSH 能力，而复制出的 Preset 可以明确启用 Codex 委派。升级 Codex 必须更新精确版本依赖并运行现有的协议、生命周期和真实产品验证套件。
