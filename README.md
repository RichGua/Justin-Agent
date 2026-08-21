# Justin Agent

本仓库是 Justin Agent 的 TypeScript 重写：Cordis 负责 Host 的可组合插件架构，官方 Codex App Server 负责代理执行、线程、审批和流式事件。

## 开发

\`\`\`powershell
corepack enable
pnpm install
pnpm --filter @justin/web dev
\`\`\`

默认仅监听回环地址。Codex 二进制由 \`JUSTIN_CODEX_BIN\` 指定；发行流程将其替换为固定的 \`@openai/codex@0.149.0\` 打包二进制。
