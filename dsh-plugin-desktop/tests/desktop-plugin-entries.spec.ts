import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Loader } from '@deepseek-ai/cordis-plugin-loader'
import {
  desktopPluginEntryStatePath,
  DesktopPluginEntriesService,
  desktopHarnessEntryOverrides,
  readDesktopPluginEntryState,
} from '../src/desktop-plugin-entries.ts'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function profileDir(): string {
  const root = mkdtempSync(join(tmpdir(), 'rundeep-plugin-entries-'))
  roots.push(root)
  return root
}

function fakeLoader() {
  const entry = (id: string, localId: string, name: string, disabled: boolean) => {
    const value = {
      id,
      options: { id: localId, name },
      disabled,
      update: vi.fn(async (options: { disabled?: boolean }) => {
        if (options.disabled !== undefined) value.disabled = options.disabled
      }),
    }
    return value
  }
  const entries = [
    entry('agent-loop', 'agent-loop', '@deepseek-ai/dsh-agent-loop', false),
    entry('codex-harness', 'codex-harness', 'dsh-plugin-desktop/codex-harness', true),
    entry('preset:tool', 'tool', '@deepseek-ai/dsh-tool', false),
  ]
  return {
    entries,
    loader: {
      *entries() { yield* entries },
    } as unknown as Pick<Loader, 'entries'>,
  }
}

function serviceFor(profile: string, primaryHarness: 'deepseek' | 'codex' = 'deepseek') {
  const harness = fakeLoader()
  return {
    harness,
    service: new DesktopPluginEntriesService({
      profileDir: profile,
      loader: harness.loader,
      primaryHarness,
    }),
  }
}

describe('Desktop harness plugin sets', () => {
  it('ships built-in harnesses and defaults every non-engine entry to the common group', () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)

    expect(service.snapshot()).toEqual({
      harnesses: [
        { id: 'deepseek', name: 'DeepSeek Harness', builtIn: true, engine: 'agent-loop', selectable: true },
        { id: 'codex', name: 'Codex Harness', builtIn: true, engine: 'codex-harness', selectable: true },
        { id: 'common', name: 'Common Plugins', builtIn: true, selectable: false },
      ],
      entries: [
        { entryId: 'agent-loop', moduleName: '@deepseek-ai/dsh-agent-loop', enabled: true, runtimeEnabled: true, harnessId: 'deepseek', engine: true, locked: true },
        { entryId: 'codex-harness', moduleName: 'dsh-plugin-desktop/codex-harness', enabled: false, runtimeEnabled: false, harnessId: 'codex', engine: true, locked: true },
        { entryId: 'preset:tool', moduleName: '@deepseek-ai/dsh-tool', enabled: true, runtimeEnabled: true, harnessId: 'common', engine: false, locked: false },
      ],
      primaryHarness: 'deepseek',
      restartRequired: false,
    })
  })

  it('links the primary harness: its engine and plugin set load, other sets stay disabled', () => {
    const profile = profileDir()
    serviceFor(profile)
    const statePath = desktopPluginEntryStatePath(profile)
    mkdirSync(join(profile, '.rundeep'))
    writeFileSync(statePath, JSON.stringify({
      version: 2,
      harnesses: [],
      entries: [
        { id: 'preset:tool', harness: 'deepseek' },
        { id: 'preset:codex-tool', harness: 'codex', enabled: true },
        { id: 'preset:common-tool', harness: 'common', enabled: false },
      ],
    }))

    expect([...desktopHarnessEntryOverrides(profile, 'deepseek')]).toEqual([
      ['agent-loop', true],
      ['codex-harness', false],
      ['preset:codex-tool', false],
      ['preset:common-tool', false],
    ])
    expect([...desktopHarnessEntryOverrides(profile, 'codex')]).toEqual([
      ['agent-loop', false],
      ['codex-harness', true],
      ['preset:tool', false],
      ['preset:common-tool', false],
    ])
  })

  it('keeps manual switches for the primary set and the common group', async () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)

    await service.setEnabled('preset:tool', false)

    expect([...desktopHarnessEntryOverrides(profile, 'deepseek')]).toEqual([
      ['agent-loop', true],
      ['codex-harness', false],
      ['preset:tool', false],
    ])
    expect(service.snapshot().restartRequired).toBe(true)
  })

  it('rejects switches on harness engine rows', async () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)

    await expect(service.setEnabled('agent-loop', false)).rejects.toThrow('harness engine entries')
    await expect(service.setEnabled('codex-harness', true)).rejects.toThrow('harness engine entries')
    await expect(service.assignHarness('agent-loop', 'common')).rejects.toThrow('engine entries cannot be reassigned')
  })

  it('creates, assigns, and deletes user harnesses without changing enablement', async () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)

    const harnessId = await service.createHarness('我的工具')
    await service.assignHarness('preset:tool', harnessId)
    const assigned = service.snapshot().entries.find(entry => entry.entryId === 'preset:tool')
    expect(assigned?.harnessId).toBe(harnessId)
    // A custom harness is not the primary set, so linking keeps it disabled.
    expect(assigned?.enabled).toBe(false)
    expect(assigned?.locked).toBe(true)

    await service.deleteHarness(harnessId)
    expect(service.snapshot().entries.find(entry => entry.entryId === 'preset:tool')?.harnessId).toBe('common')
  })

  it('refuses duplicate harness names and owned engine rows', async () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)

    await service.createHarness('claude')
    await expect(service.createHarness('Claude')).rejects.toThrow('harness already exists')
    await expect(service.createHarness('claude-2', 'agent-loop')).rejects.toThrow('already owned')
    await expect(service.createHarness('claude-2', 'missing-row')).rejects.toThrow('unavailable')
  })

  it('refuses deleting built-in harnesses and releases custom entries to common', async () => {
    const profile = profileDir()
    const { service } = serviceFor(profile)
    const harnessId = await service.createHarness('custom')

    await expect(service.deleteHarness('deepseek')).rejects.toThrow('built-in harnesses cannot be deleted')
    await expect(service.deleteHarness('codex')).rejects.toThrow('built-in harnesses cannot be deleted')
    await expect(service.deleteHarness(harnessId)).resolves.toBeUndefined()
  })

  it('migrates legacy v1 category state to v2 harness state', async () => {
    const profile = profileDir()
    const statePath = desktopPluginEntryStatePath(profile)
    mkdirSync(join(profile, '.rundeep'))
    writeFileSync(statePath, JSON.stringify({
      version: 1,
      entries: [
        { id: 'agent-loop', category: 'deepseek' },
        { id: 'codex-harness', category: 'codex', enabled: true },
        { id: 'preset:tool', enabled: false },
      ],
      categories: [{ id: `custom_${'a'.repeat(32)}`, name: '我的工具' }],
    }))

    const state = readDesktopPluginEntryState(profile)
    expect(state.version).toBe(2)
    expect(state.harnesses).toEqual([{ id: `custom_${'a'.repeat(32)}`, name: '我的工具' }])
    expect(state.entries).toEqual([
      { id: 'agent-loop', harness: 'deepseek' },
      { id: 'codex-harness', harness: 'codex', enabled: true },
      { id: 'preset:tool', enabled: false },
    ])

    const { service } = serviceFor(profile)
    await service.setEnabled('preset:tool', true)
    expect(JSON.parse(readFileSync(statePath, 'utf8')).version).toBe(2)
  })

  it('reconciles persisted nested Include entries after their Loader subtree mounts', async () => {
    const profile = profileDir()
    const { service, harness } = serviceFor(profile)
    await service.setEnabled('preset:tool', false)

    await service.reconcile()

    expect(harness.entries[2]?.update).toHaveBeenCalledWith({ disabled: true })
    expect(harness.entries[2]?.disabled).toBe(true)
  })

  it('rejects malformed or oversized profile state instead of guessing', () => {
    const profile = profileDir()
    const statePath = desktopPluginEntryStatePath(profile)
    mkdirSync(join(profile, '.rundeep'))
    writeFileSync(statePath, JSON.stringify({ version: 2, harnesses: [], entries: [{ id: 'bad id', enabled: true }] }))

    expect(() => readDesktopPluginEntryState(profile)).toThrow('invalid plugin entry state')
    expect(readFileSync(statePath, 'utf8')).toContain('bad id')
  })
})
