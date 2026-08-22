import { describe, expect, it } from 'vitest'
import {
  CODEX_HARNESS_ENGINE,
  CODEX_HARNESS_PLUGIN,
  COMMON_HARNESS_ID,
  DEEPSEEK_HARNESS_ENGINE,
  DEEPSEEK_HARNESS_FOUNDATION,
  DEEPSEEK_HARNESS_PLUGIN,
  HARNESS_PROVIDERS,
  harnessProviderForEngine,
  harnessProviderForPlugin,
} from '../src/harnesses.ts'

describe('Harness provider classification', () => {
  it('ships exactly the two built-in primary harnesses plus the common group', () => {
    expect(HARNESS_PROVIDERS.map(provider => [provider.id, provider.builtIn, provider.engine])).toEqual([
      ['deepseek', true, 'agent-loop'],
      ['codex', true, 'codex-harness'],
    ])
    expect(COMMON_HARNESS_ID).toBe('common')
    expect(DEEPSEEK_HARNESS_ENGINE).toBe('agent-loop')
    expect(CODEX_HARNESS_ENGINE).toBe('codex-harness')
  })

  it('keeps the Codex SDK adapter distinct from the DeepSeek Codex subagent', () => {
    expect(DEEPSEEK_HARNESS_FOUNDATION).toBe('@deepseek-ai/dsh-base')
    expect(harnessProviderForPlugin(DEEPSEEK_HARNESS_PLUGIN)?.id).toBe('deepseek')
    expect(harnessProviderForPlugin(CODEX_HARNESS_PLUGIN)?.id).toBe('codex')
    expect(harnessProviderForEngine('agent-loop')?.id).toBe('deepseek')
    expect(harnessProviderForEngine('codex-harness')?.id).toBe('codex')
    expect(CODEX_HARNESS_PLUGIN).toBe('dsh-plugin-desktop/codex-harness')
    expect(harnessProviderForPlugin('@deepseek-ai/dsh-subagent-codex')).toBeUndefined()
    expect(harnessProviderForPlugin('@deepseek-ai/dsh-web-app')).toBeUndefined()
    expect(harnessProviderForEngine('third-party-engine')).toBeUndefined()
  })
})
