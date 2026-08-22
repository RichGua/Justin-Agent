/** Stable primary AgentFactory classification for Desktop profiles. */

/** One family of agent runtime that can own an enabled Profile bundle. */
export type HarnessKind = 'deepseek' | 'codex' | 'claude'

/** Declarative metadata for plugin-center and Profile inventory clients. */
export interface HarnessProvider {
  readonly id: HarnessKind
  readonly label: string
  readonly plugin: string
  readonly builtIn: boolean
  readonly exclusiveGroup: 'primary-agent'
}

/** DeepSeek Harness remains the foundation for every provider. */
export const DEEPSEEK_HARNESS_FOUNDATION = '@deepseek-ai/dsh-base'

/** Upstream DeepSeek Harness AgentFactory implementation. */
export const DEEPSEEK_HARNESS_PLUGIN = '@deepseek-ai/dsh-agent-loop'

/**
 * Desktop-owned AgentFactory backed by the official `@openai/codex-sdk`.
 */
export const CODEX_HARNESS_PLUGIN = 'dsh-plugin-desktop/codex-harness'

/** Provider bundles are alternatives, not models added to one agent loop. */
export const HARNESS_PROVIDERS: readonly HarnessProvider[] = Object.freeze([
  { id: 'deepseek', label: 'DeepSeek Harness', plugin: DEEPSEEK_HARNESS_PLUGIN, builtIn: true, exclusiveGroup: 'primary-agent' },
  { id: 'codex', label: 'Codex Harness', plugin: CODEX_HARNESS_PLUGIN, builtIn: true, exclusiveGroup: 'primary-agent' },
  { id: 'claude', label: 'Anthropic Claude', plugin: 'dsh-harness-claude', builtIn: false, exclusiveGroup: 'primary-agent' },
])

/** Return the provider declared by one exact Loader plugin identity. */
export function harnessProviderForPlugin(plugin: string): HarnessProvider | undefined {
  return HARNESS_PROVIDERS.find(provider => provider.plugin === plugin)
}
