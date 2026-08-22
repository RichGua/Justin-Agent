/** Stable primary AgentFactory classification for Desktop profiles. */

/** Loader row id of the upstream DeepSeek Harness AgentFactory. */
export const DEEPSEEK_HARNESS_ENGINE = 'agent-loop'
/** Loader row id of the Desktop-owned Codex AgentFactory. */
export const CODEX_HARNESS_ENGINE = 'codex-harness'

/** Upstream DeepSeek Harness AgentFactory package identity. */
export const DEEPSEEK_HARNESS_PLUGIN = '@deepseek-ai/dsh-agent-loop'
/** Desktop-owned Codex AgentFactory package identity. */
export const CODEX_HARNESS_PLUGIN = 'dsh-plugin-desktop/codex-harness'
/** DeepSeek Harness remains the foundation for every provider. */
export const DEEPSEEK_HARNESS_FOUNDATION = '@deepseek-ai/dsh-base'

/** Built-in primary harness id for the upstream DeepSeek AgentFactory. */
export const DEEPSEEK_HARNESS_ID = 'deepseek'
/** Built-in primary harness id for the Desktop-owned Codex AgentFactory. */
export const CODEX_HARNESS_ID = 'codex'
/** Generic group id for plugins that no harness owns; it never drives composition. */
export const COMMON_HARNESS_ID = 'common'

/** Ids of the two built-in primary harnesses. */
export type HarnessKind = typeof DEEPSEEK_HARNESS_ID | typeof CODEX_HARNESS_ID
/** Ids of user-created harnesses; the custom prefix is stable across profiles. */
export type CustomHarnessId = `custom_${string}`
/** Every harness or generic group id used by plugin assignment. */
export type HarnessId = HarnessKind | typeof COMMON_HARNESS_ID | CustomHarnessId
/** Harness ids that can own the primary AgentFactory selection. */
export type PrimaryHarnessId = HarnessKind | CustomHarnessId

/** User-created harness id pattern shared by state, settings, and the client. */
export const CUSTOM_HARNESS_ID_PATTERN = /^custom_[a-f0-9]{32}$/u

/** One built-in harness definition used by state defaults and composition. */
export interface HarnessProvider {
  /** Stable built-in harness id. */
  readonly id: HarnessKind
  /** Stable display name; the client localizes built-in ids. */
  readonly name: string
  /** Loader row id that supplies the AgentFactory owned by this harness. */
  readonly engine: string
  /** Package identity of the AgentFactory plugin. */
  readonly plugin: string
  /** Built-in harnesses are always present and never deletable. */
  readonly builtIn: true
}

/** Built-in harness definitions; user harnesses live in profile state. */
export const HARNESS_PROVIDERS: readonly HarnessProvider[] = Object.freeze([
  {
    id: DEEPSEEK_HARNESS_ID,
    name: 'DeepSeek Harness',
    engine: DEEPSEEK_HARNESS_ENGINE,
    plugin: DEEPSEEK_HARNESS_PLUGIN,
    builtIn: true,
  },
  {
    id: CODEX_HARNESS_ID,
    name: 'Codex Harness',
    engine: CODEX_HARNESS_ENGINE,
    plugin: CODEX_HARNESS_PLUGIN,
    builtIn: true,
  },
])

/** Return the built-in provider that owns one exact engine Loader row id. */
export function harnessProviderForEngine(engine: string): HarnessProvider | undefined {
  return HARNESS_PROVIDERS.find(provider => provider.engine === engine)
}

/** Return the built-in provider declared by one exact AgentFactory package identity. */
export function harnessProviderForPlugin(plugin: string): HarnessProvider | undefined {
  return HARNESS_PROVIDERS.find(provider => provider.plugin === plugin)
}
