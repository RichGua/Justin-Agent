import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Loader } from '@deepseek-ai/cordis-plugin-loader'
import {
  desktopPluginEntryStatePath,
  DesktopPluginEntriesService,
  readDesktopPluginEntryOverrides,
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

describe('Desktop Loader plugin entries', () => {
  it('starts with only DeepSeek and Codex categories and exposes every Loader row', () => {
    const profile = profileDir()
    const harness = fakeLoader()
    const service = new DesktopPluginEntriesService({ profileDir: profile, loader: harness.loader })

    expect(service.snapshot()).toEqual({
      categories: [
        { id: 'deepseek', name: 'DeepSeek Harness', builtIn: true },
        { id: 'codex', name: 'Codex Harness', builtIn: true },
      ],
      entries: [
        { entryId: 'agent-loop', moduleName: '@deepseek-ai/dsh-agent-loop', enabled: true, runtimeEnabled: true, categoryId: 'deepseek' },
        { entryId: 'codex-harness', moduleName: 'dsh-plugin-desktop/codex-harness', enabled: false, runtimeEnabled: false, categoryId: 'codex' },
        { entryId: 'preset:tool', moduleName: '@deepseek-ai/dsh-tool', enabled: true, runtimeEnabled: true, categoryId: 'deepseek' },
      ],
      restartRequired: false,
    })
  })

  it('persists official id/disabled intent and keeps primary Harness entries exclusive', async () => {
    const profile = profileDir()
    const harness = fakeLoader()
    const service = new DesktopPluginEntriesService({ profileDir: profile, loader: harness.loader })

    await service.setEnabled('codex-harness', true)

    expect([...readDesktopPluginEntryOverrides(profile)]).toEqual([
      ['codex-harness', true],
      ['agent-loop', false],
    ])
    expect(service.snapshot()).toEqual(expect.objectContaining({ restartRequired: true }))
  })

  it('creates, assigns, and deletes user categories without changing enablement', async () => {
    const profile = profileDir()
    const harness = fakeLoader()
    const service = new DesktopPluginEntriesService({ profileDir: profile, loader: harness.loader })

    const categoryId = await service.createCategory('我的工具')
    await service.assignCategory('preset:tool', categoryId)
    expect(service.snapshot().entries.find(entry => entry.entryId === 'preset:tool')?.categoryId).toBe(categoryId)
    expect(service.snapshot().restartRequired).toBe(false)

    await service.deleteCategory(categoryId)
    expect(service.snapshot().entries.find(entry => entry.entryId === 'preset:tool')?.categoryId).toBe('deepseek')
  })

  it('reconciles persisted nested Include entries after their Loader subtree mounts', async () => {
    const profile = profileDir()
    const harness = fakeLoader()
    const service = new DesktopPluginEntriesService({ profileDir: profile, loader: harness.loader })
    await service.setEnabled('preset:tool', false)

    await service.reconcile()

    expect(harness.entries[2]?.update).toHaveBeenCalledWith({ disabled: true })
    expect(harness.entries[2]?.disabled).toBe(true)
  })

  it('rejects malformed or oversized profile state instead of guessing', () => {
    const profile = profileDir()
    const statePath = desktopPluginEntryStatePath(profile)
    mkdirSync(join(profile, '.rundeep'))
    writeFileSync(statePath, JSON.stringify({ version: 1, entries: [{ id: 'bad id', enabled: true }], categories: [] }))

    expect(() => readDesktopPluginEntryOverrides(profile)).toThrow('invalid plugin entry state')
    expect(readFileSync(statePath, 'utf8')).toContain('bad id')
  })
})
