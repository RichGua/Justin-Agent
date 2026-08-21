import { describe, expect, it } from 'vitest'
import { HARNESS_PROVIDERS, harnessProviderForBundle } from '../src/harnesses.ts'

describe('Harness provider classification', () => {
  it('makes DeepSeek built in and alternative harnesses opt-in', () => {
    expect(HARNESS_PROVIDERS.map(provider => [provider.id, provider.builtIn])).toEqual([
      ['deepseek', true], ['codex', false], ['claude', false],
    ])
  })

  it('classifies exact provider bundles only', () => {
    expect(harnessProviderForBundle('@justin-agent/dsh-harness-codex')?.id).toBe('codex')
    expect(harnessProviderForBundle('@deepseek-ai/dsh-web-app')).toBeUndefined()
  })
})
