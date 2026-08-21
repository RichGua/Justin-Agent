# Agent Note: Ship Codex App Server as an opt-in Harness bundle

Status: implemented

English | [中文](2026-08-22-codex-app-server-bundle.zh.md)

## Problem

Justin needs DeepSeek Harness's replaceable Cordis product structure while using the official Codex App Server as an optional execution product. A machine `codex` executable is not a reliable compatibility boundary because its version and protocol can drift independently of the Harness.

## Decision

`@deepseek-ai/dsh-subagent-codex` is the in-repository Codex product bundle. It pins `@openai/codex` to `0.149.0`, resolves that package's own executable, and speaks only JSON-RPC over `app-server --stdio`. The provider remains dormant when mounted: it starts no process until an enabled agent-preset delegation tool invokes it.

The web profile template contains the bundle, so a fresh local profile has the provider available without a separate registry install. The standard preset keeps `subagent_codex` disabled. A user turns it on by copying the preset and enabling that one tool row; removing the bundle from the profile withdraws the provider on the next restart. This keeps product availability and agent permission separate.

## Alternatives considered

**A second handwritten Codex agent loop** — rejected. The App Server owns thread, turn, approval, sandbox, account, model, MCP, skill, and native plugin behavior; duplicating that loop would inevitably diverge.

**Use the `codex` executable found on `PATH`** — rejected. It would silently change the protocol baseline between machines. The package-local `0.149.0` wrapper selects the matching native payload instead.

**Enable the delegation tool in every new session** — rejected. Mounting a provider must not grant its model-facing tool or launch work without the user selecting it.

## Consequences

Fresh Web profiles include the Codex product capability but do not create an App Server process or expose it to an agent by default. The browser surface retains all native DSH functionality, while a copied preset can explicitly enable Codex delegation. Updating Codex requires the exact-version dependency change and the existing protocol, lifecycle, and real-product verification suite.
