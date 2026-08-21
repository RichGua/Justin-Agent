/** Stable Harness-provider classification for Desktop profiles. */

/** One family of agent runtime that can own an enabled Profile bundle. */
export type HarnessKind = 'deepseek' | 'codex' | 'claude'

/** Declarative metadata for plugin-center and Profile inventory clients. */
export interface HarnessProvider {
  readonly id: HarnessKind
  readonly label: string
  readonly bundle: string
  readonly builtIn: boolean
  readonly exclusiveGroup: 'primary-agent'
}

/** Provider bundles are alternatives, not models added to one agent loop. */
export const HARNESS_PROVIDERS: readonly HarnessProvider[] = Object.freeze([
  { id: 'deepseek', label: 'DeepSeek Harness', bundle: '@deepseek-ai/dsh-base', builtIn: true, exclusiveGroup: 'primary-agent' },
  { id: 'codex', label: 'OpenAI Codex', bundle: '@justin-agent/dsh-harness-codex', builtIn: false, exclusiveGroup: 'primary-agent' },
  { id: 'claude', label: 'Anthropic Claude', bundle: '@justin-agent/dsh-harness-claude', builtIn: false, exclusiveGroup: 'primary-agent' },
])

/** Return the provider declared by one exact bundle identity. */
export function harnessProviderForBundle(bundle: string): HarnessProvider | undefined {
  return HARNESS_PROVIDERS.find(provider => provider.bundle === bundle)
}
