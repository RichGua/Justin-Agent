import { describe, expect, it } from 'vitest'
import {
  CODEX_HARNESS_PLUGIN,
  DEEPSEEK_HARNESS_FOUNDATION,
  DEEPSEEK_HARNESS_PLUGIN,
  HARNESS_PROVIDERS,
  harnessProviderForPlugin,
} from '../src/harnesses.ts'

describe('Harness provider classification', () => {
  it('ships DeepSeek and Codex while keeping future harnesses opt-in', () => {
    expect(HARNESS_PROVIDERS.map(provider => [provider.id, provider.builtIn])).toEqual([
      ['deepseek', true], ['codex', true], ['claude', false],
    ])
  })

  it('keeps the Codex SDK adapter distinct from the DeepSeek Codex subagent', () => {
    expect(DEEPSEEK_HARNESS_FOUNDATION).toBe('@deepseek-ai/dsh-base')
    expect(harnessProviderForPlugin(DEEPSEEK_HARNESS_PLUGIN)?.id).toBe('deepseek')
    expect(harnessProviderForPlugin(CODEX_HARNESS_PLUGIN)?.id).toBe('codex')
    expect(CODEX_HARNESS_PLUGIN).toBe('dsh-plugin-desktop/codex-harness')
    expect(harnessProviderForPlugin('@deepseek-ai/dsh-subagent-codex')).toBeUndefined()
    expect(harnessProviderForPlugin('@deepseek-ai/dsh-web-app')).toBeUndefined()
  })
})
