// @vitest-environment jsdom
import { createElement } from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DesktopPluginsView, DesktopSettingsApi } from '../src/client/desktop-settings-api.ts'
import { HarnessPluginInventory, pluginFamilyOf } from '../src/client/plugin-inventory.tsx'

afterEach(cleanup)

const plugins: DesktopPluginsView = {
  bundles: [],
  categories: [
    { id: 'deepseek', name: 'DeepSeek Harness', builtIn: true },
    { id: 'codex', name: 'Codex Harness', builtIn: true },
  ],
  entries: [
    { entryId: 'agent-loop', moduleName: '@deepseek-ai/dsh-agent-loop', enabled: true, categoryId: 'deepseek' },
    { entryId: 'subagent-codex', moduleName: '@deepseek-ai/dsh-subagent-codex', enabled: false, categoryId: 'deepseek' },
    { entryId: 'codex-harness', moduleName: 'dsh-plugin-desktop/codex-harness', enabled: false, categoryId: 'codex' },
  ],
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
    createPluginCategory: vi.fn(),
    assignPluginCategory: vi.fn(),
    deletePluginCategory: vi.fn(),
    restartPlugins: vi.fn(async () => ({ accepted: true as const, restartRequired: true })),
    ...overrides,
  } satisfies Pick<DesktopSettingsApi,
    'readPlugins' | 'setPluginEntryEnabled' | 'createPluginCategory' | 'assignPluginCategory' | 'deletePluginCategory' | 'restartPlugins'>
}

describe('RunDeep Harness plugin inventory', () => {
  it('uses only the Codex SDK adapter for the default Codex classification', () => {
    expect(pluginFamilyOf('dsh-plugin-desktop/codex-harness')).toBe('codex')
    expect(pluginFamilyOf('@deepseek-ai/dsh-subagent-codex')).toBe('deepseek')
    expect(pluginFamilyOf('third-party-plugin')).toBe('deepseek')
  })

  it('shows exactly the two default categories and a real switch on every Loader row', async () => {
    const client = api()
    render(createElement(HarnessPluginInventory, { api: client, t: (key: string) => key } as never))

    expect(await screen.findByRole('button', { name: 'collapse: DeepSeek Harness' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'collapse: Codex Harness' })).toBeTruthy()
    expect(screen.getAllByRole('switch')).toHaveLength(3)
    expect(screen.queryByText(/immutable|managed/u)).toBeNull()

    fireEvent.click(screen.getByRole('switch', { name: 'enable: codex-harness' }))
    await waitFor(() => {
      expect(client.setPluginEntryEnabled).toHaveBeenCalledWith('codex-harness', true)
      expect(screen.getByRole('switch', { name: 'disable: codex-harness' })).toBeTruthy()
    })
    expect(screen.getByRole('button', { name: 'restart' })).toBeTruthy()
  })

  it('creates a user category and assigns an entry without inventing a third default group', async () => {
    const category = { id: `custom_${'a'.repeat(32)}`, name: '我的工具', builtIn: false }
    const withCategory: DesktopPluginsView = { ...plugins, categories: [...plugins.categories, category] }
    const assigned: DesktopPluginsView = {
      ...withCategory,
      entries: withCategory.entries.map(entry => entry.entryId === 'subagent-codex'
        ? { ...entry, categoryId: category.id }
        : entry),
    }
    const client = api({
      createPluginCategory: vi.fn(async () => withCategory),
      assignPluginCategory: vi.fn(async () => assigned),
      deletePluginCategory: vi.fn(async () => plugins),
    })
    render(createElement(HarnessPluginInventory, { api: client, t: (key: string) => key } as never))

    fireEvent.change(await screen.findByPlaceholderText('categoryName'), { target: { value: '我的工具' } })
    fireEvent.click(screen.getByRole('button', { name: 'createCategory' }))
    expect(await screen.findByRole('button', { name: 'collapse: 我的工具' })).toBeTruthy()

    const categorySelect = screen.getByLabelText('category: subagent-codex')
    fireEvent.change(categorySelect, { target: { value: category.id } })
    await waitFor(() => { expect(client.assignPluginCategory).toHaveBeenCalledWith('subagent-codex', category.id) })
    expect(screen.getByRole('button', { name: 'deleteCategory: 我的工具' })).toBeTruthy()
  })
})
