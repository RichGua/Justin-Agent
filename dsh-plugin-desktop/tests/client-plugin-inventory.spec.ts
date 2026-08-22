// @vitest-environment jsdom
import { createElement } from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SettingsScope } from '@deepseek-ai/dsh-client-runtime/client'
import type { DesktopPluginsView, DesktopSettingsApi } from '../src/client/desktop-settings-api.ts'
import type { DesktopShellSettings } from '../src/client/DesktopSettingsSection.tsx'
import { HarnessPluginInventory } from '../src/client/plugin-inventory.tsx'

afterEach(cleanup)

const plugins: DesktopPluginsView = {
  bundles: [],
  harnesses: [
    { id: 'deepseek', name: 'DeepSeek Harness', builtIn: true, engine: 'agent-loop', selectable: true },
    { id: 'codex', name: 'Codex Harness', builtIn: true, engine: 'codex-harness', selectable: true },
    { id: 'common', name: 'Common Plugins', builtIn: true, selectable: false },
  ],
  entries: [
    { entryId: 'agent-loop', moduleName: '@deepseek-ai/dsh-agent-loop', enabled: true, harnessId: 'deepseek', engine: true, locked: true },
    { entryId: 'subagent-codex', moduleName: '@deepseek-ai/dsh-subagent-codex', enabled: false, harnessId: 'common', engine: false, locked: false },
    { entryId: 'codex-harness', moduleName: 'dsh-plugin-desktop/codex-harness', enabled: false, harnessId: 'codex', engine: true, locked: true },
  ],
  primaryHarness: 'deepseek',
  restartRequired: false,
}

function api(overrides: Partial<DesktopSettingsApi> = {}) {
  return {
    readPlugins: vi.fn(async () => plugins),
    setPluginEntryEnabled: vi.fn(async (entryId: string, enabled: boolean) => ({
      ...plugins,
      entries: plugins.entries.map(entry => entry.entryId === entryId ? { ...entry, enabled } : entry),
      restartRequired: true,
    })),
    createHarness: vi.fn(),
    assignHarness: vi.fn(),
    deleteHarness: vi.fn(),
    restartPlugins: vi.fn(async () => ({ accepted: true as const, restartRequired: true })),
    ...overrides,
  } satisfies Pick<DesktopSettingsApi,
    'readPlugins' | 'setPluginEntryEnabled' | 'createHarness' | 'assignHarness' | 'deleteHarness' | 'restartPlugins'>
}

function settingsScope() {
  return {
    set: vi.fn(async () => {}),
    getSnapshot: () => ({
      status: 'ready' as const,
      value: { mode: 'compatibility' as const, port: 0, harness: 'deepseek', logLevel: 'info' as const },
      writable: true,
    }),
    subscribe: () => () => {},
  } as unknown as SettingsScope<DesktopShellSettings>
}

function renderInventory(client: ReturnType<typeof api>, desktopSettings = settingsScope()) {
  return render(createElement(HarnessPluginInventory, {
    api: client,
    desktopSettings,
    t: (key: string) => key,
  } as never))
}

describe('Rundeep Harness plugin inventory', () => {
  it('shows the three built-in groups and a segmented engine switch', async () => {
    const client = api()
    renderInventory(client)

    expect(await screen.findByRole('button', { name: 'collapse: DeepSeek Harness' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'collapse: Codex Harness' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'collapse: common' })).toBeTruthy()
    expect(screen.getByText('primary')).toBeTruthy()
    // The engine switch is a segmented radio group with exactly the selectable harnesses.
    expect(screen.getByRole('radio', { name: 'DeepSeek Harness' }).getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('radio', { name: 'Codex Harness' }).getAttribute('aria-checked')).toBe('false')
    // Engine rows render but their switches are locked by the harness setting.
    const switches = screen.getAllByRole('switch')
    expect(switches).toHaveLength(3)
    expect((switches[0] as HTMLButtonElement).disabled).toBe(true)
    expect((switches[1] as HTMLButtonElement).disabled).toBe(true)
    expect((switches[2] as HTMLButtonElement).disabled).toBe(false)
    expect(screen.getByRole('switch', { name: 'enable: subagent-codex' })).toBeTruthy()
  })

  it('switches the primary harness through the settings scope', async () => {
    const client = api()
    const desktopSettings = settingsScope()
    renderInventory(client, desktopSettings)

    fireEvent.click(await screen.findByRole('radio', { name: 'Codex Harness' }))
    await waitFor(() => { expect(desktopSettings.set).toHaveBeenCalledWith('harness', 'codex') })
  })

  it('toggles a common-group entry and reports a restart', async () => {
    const client = api()
    renderInventory(client)

    fireEvent.click(await screen.findByRole('switch', { name: 'enable: subagent-codex' }))
    await waitFor(() => {
      expect(client.setPluginEntryEnabled).toHaveBeenCalledWith('subagent-codex', true)
      expect(screen.getByRole('switch', { name: 'disable: subagent-codex' })).toBeTruthy()
    })
    expect(screen.getByRole('button', { name: 'restart' })).toBeTruthy()
  })

  it('moves an entry to a user harness through the row menu', async () => {
    const harness = { id: `custom_${'a'.repeat(32)}`, name: '我的工具', builtIn: false, selectable: false }
    const withHarness: DesktopPluginsView = { ...plugins, harnesses: [...plugins.harnesses, harness] }
    const assigned: DesktopPluginsView = {
      ...withHarness,
      entries: withHarness.entries.map(entry => entry.entryId === 'subagent-codex'
        ? { ...entry, harnessId: harness.id, locked: true }
        : entry),
    }
    const client = api({
      createHarness: vi.fn(async () => withHarness),
      assignHarness: vi.fn(async () => assigned),
      deleteHarness: vi.fn(async () => plugins),
    })
    renderInventory(client)

    fireEvent.change(await screen.findByPlaceholderText('harnessName'), { target: { value: '我的工具' } })
    fireEvent.click(screen.getByRole('button', { name: 'createHarness' }))
    expect(await screen.findByRole('button', { name: 'collapse: 我的工具' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'move: subagent-codex' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: '我的工具' }))
    await waitFor(() => { expect(client.assignHarness).toHaveBeenCalledWith('subagent-codex', harness.id) })
    expect(screen.getByRole('button', { name: 'deleteHarness: 我的工具' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'deleteHarness: 我的工具' }))
    await waitFor(() => { expect(client.deleteHarness).toHaveBeenCalledWith(harness.id) })
  })
})
