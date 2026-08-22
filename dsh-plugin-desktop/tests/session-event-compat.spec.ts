import { KNOWN_SESSION_EVENT_TYPES } from '@deepseek-ai/dsh-session'
import { describe, expect, it } from 'vitest'
import {
  LEGACY_CODEX_SESSION_EVENT_TYPES,
  installLegacyCodexSessionEventCompatibility,
} from '../src/session-event-compat.ts'

describe('legacy Codex session event compatibility', () => {
  it('admits only the historical informational Desktop event vocabulary', () => {
    const unrelated = 'desktop-test/unrelated-required-event'
    expect(KNOWN_SESSION_EVENT_TYPES.has(unrelated)).toBe(false)

    installLegacyCodexSessionEventCompatibility()

    expect(LEGACY_CODEX_SESSION_EVENT_TYPES).toEqual(['codex/thread', 'codex/item'])
    expect(KNOWN_SESSION_EVENT_TYPES.has('codex/thread')).toBe(true)
    expect(KNOWN_SESSION_EVENT_TYPES.has('codex/item')).toBe(true)
    expect(KNOWN_SESSION_EVENT_TYPES.has(unrelated)).toBe(false)
  })
})
