/** Profile-scoped management for the effective Cordis Loader plugin tree. */

import { randomBytes } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { chmod, lstat, mkdir } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'
import type { Loader } from '@deepseek-ai/cordis-plugin-loader'
import { withFileLock, writeFileAtomic } from '@deepseek-ai/dsh-atomic-write'

const BIN_NAME = 'dsh-plugin-desktop'
const STATE_VERSION = 1
const STATE_FILE = join('.rundeep', 'plugins.json')
const STATE_FILE_MODE = 0o600
const STATE_DIRECTORY_MODE = 0o700
const MAX_STATE_BYTES = 256 * 1024
const MAX_ENTRIES = 2048
const MAX_CATEGORIES = 64
const MAX_CATEGORY_NAME_LENGTH = 64
const ENTRY_ID_PATTERN = /^[A-Za-z0-9._:@/-]{1,256}$/u
const CUSTOM_CATEGORY_ID_PATTERN = /^custom_[a-f0-9]{32}$/u
const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/u

export const DEEPSEEK_PLUGIN_CATEGORY_ID = 'deepseek'
export const CODEX_PLUGIN_CATEGORY_ID = 'codex'
export const CODEX_HARNESS_ENTRY_ID = 'codex-harness'
export const DEEPSEEK_HARNESS_ENTRY_ID = 'agent-loop'

interface StoredPluginEntry {
  readonly id: string
  readonly enabled?: boolean
  readonly category?: string
}

interface StoredPluginCategory {
  readonly id: string
  readonly name: string
}

export interface DesktopPluginEntryState {
  readonly version: 1
  readonly entries: readonly StoredPluginEntry[]
  readonly categories: readonly StoredPluginCategory[]
}

export interface DesktopPluginEntryView {
  readonly entryId: string
  readonly moduleName: string
  /** Persisted state used by the next generation. */
  readonly enabled: boolean
  /** Effective state in the currently running Loader tree. */
  readonly runtimeEnabled: boolean
  readonly categoryId: string
}

export interface DesktopPluginCategoryView {
  readonly id: string
  readonly name: string
  readonly builtIn: boolean
}

export interface DesktopPluginEntrySnapshot {
  readonly entries: readonly DesktopPluginEntryView[]
  readonly categories: readonly DesktopPluginCategoryView[]
  readonly restartRequired: boolean
}

export interface DesktopPluginEntriesBootstrap {
  readonly profileDir: string
  readonly loader: Pick<Loader, 'entries'>
}

function emptyState(): DesktopPluginEntryState {
  return { version: STATE_VERSION, entries: [], categories: [] }
}

function isExactKeys(value: Record<string, unknown>, allowed: readonly string[]): boolean {
  const keys = Object.keys(value)
  return keys.every(key => allowed.includes(key))
}

function parseCategoryName(value: unknown): string {
  if (typeof value !== 'string') throw new Error('category name must be a string')
  const name = value.trim()
  if (name.length === 0 || name.length > MAX_CATEGORY_NAME_LENGTH
    || CONTROL_CHARACTER_PATTERN.test(name)) {
    throw new Error('category name is invalid')
  }
  return name
}

function parseState(value: unknown): DesktopPluginEntryState {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('state must be an object')
  }
  const record = value as Record<string, unknown>
  if (!isExactKeys(record, ['version', 'entries', 'categories'])
    || record.version !== STATE_VERSION
    || !Array.isArray(record.entries)
    || !Array.isArray(record.categories)
    || record.entries.length > MAX_ENTRIES
    || record.categories.length > MAX_CATEGORIES) {
    throw new Error('state shape is invalid')
  }

  const categories: StoredPluginCategory[] = []
  const categoryIds = new Set<string>()
  const categoryNames = new Set<string>(['deepseek harness', 'codex harness'])
  for (const raw of record.categories) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error('category is invalid')
    }
    const category = raw as Record<string, unknown>
    if (!isExactKeys(category, ['id', 'name'])
      || typeof category.id !== 'string'
      || !CUSTOM_CATEGORY_ID_PATTERN.test(category.id)) {
      throw new Error('category is invalid')
    }
    const name = parseCategoryName(category.name)
    const normalizedName = name.toLocaleLowerCase()
    if (categoryIds.has(category.id) || categoryNames.has(normalizedName)) {
      throw new Error('category is duplicated')
    }
    categoryIds.add(category.id)
    categoryNames.add(normalizedName)
    categories.push({ id: category.id, name })
  }

  const entries: StoredPluginEntry[] = []
  const entryIds = new Set<string>()
  for (const raw of record.entries) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error('entry state is invalid')
    }
    const entry = raw as Record<string, unknown>
    if (!isExactKeys(entry, ['id', 'enabled', 'category'])
      || typeof entry.id !== 'string'
      || !ENTRY_ID_PATTERN.test(entry.id)
      || (entry.enabled !== undefined && typeof entry.enabled !== 'boolean')
      || (entry.category !== undefined
        && (typeof entry.category !== 'string'
          || (!categoryIds.has(entry.category)
            && entry.category !== DEEPSEEK_PLUGIN_CATEGORY_ID
            && entry.category !== CODEX_PLUGIN_CATEGORY_ID)))
      || (entry.enabled === undefined && entry.category === undefined)
      || entryIds.has(entry.id)) {
      throw new Error('entry state is invalid')
    }
    entryIds.add(entry.id)
    entries.push({
      id: entry.id,
      ...(entry.enabled === undefined ? {} : { enabled: entry.enabled }),
      ...(entry.category === undefined ? {} : { category: entry.category }),
    })
  }
  return { version: STATE_VERSION, entries, categories }
}

function renderState(state: DesktopPluginEntryState): string {
  return `${JSON.stringify(state, null, 2)}\n`
}

/** Resolve the single launcher-managed state file inside one Profile. */
export function desktopPluginEntryStatePath(profileDir: string): string {
  if (!isAbsolute(profileDir) || profileDir.includes('\0')) {
    throw new Error(`${BIN_NAME}: plugin entry profile path must be absolute and contain no NUL`)
  }
  return join(profileDir, STATE_FILE)
}

/** Strictly read the state used by both pre-boot composition and the Host API. */
export function readDesktopPluginEntryState(profileDir: string): DesktopPluginEntryState {
  const statePath = desktopPluginEntryStatePath(profileDir)
  if (!existsSync(statePath)) return emptyState()
  try {
    const content = readFileSync(statePath, 'utf8')
    if (Buffer.byteLength(content, 'utf8') > MAX_STATE_BYTES) {
      throw new Error('state is too large')
    }
    return parseState(JSON.parse(content) as unknown)
  } catch (cause) {
    throw new Error(
      `${BIN_NAME}: invalid plugin entry state at ${statePath}: ${cause instanceof Error ? cause.message : String(cause)}`,
    )
  }
}

/** Read only persisted enabled overrides for official Cordis patch composition. */
export function readDesktopPluginEntryOverrides(profileDir: string): ReadonlyMap<string, boolean> {
  return new Map(
    readDesktopPluginEntryState(profileDir).entries
      .filter((entry): entry is StoredPluginEntry & { enabled: boolean } => entry.enabled !== undefined)
      .map(entry => [entry.id, entry.enabled]),
  )
}

function defaultCategory(entryId: string): string {
  return entryId === CODEX_HARNESS_ENTRY_ID
    ? CODEX_PLUGIN_CATEGORY_ID
    : DEEPSEEK_PLUGIN_CATEGORY_ID
}

async function ensurePrivateStateDirectory(statePath: string): Promise<void> {
  const directory = dirname(statePath)
  await mkdir(directory, { recursive: true, mode: STATE_DIRECTORY_MODE })
  const stat = await lstat(directory)
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error(`${BIN_NAME}: plugin entry state directory is not private`)
  }
  await chmod(directory, STATE_DIRECTORY_MODE)
}

/** Persistent category and enablement mutations over the effective Loader tree. */
export class DesktopPluginEntriesService {
  private readonly statePath: string

  constructor(private readonly bootstrap: DesktopPluginEntriesBootstrap) {
    this.statePath = desktopPluginEntryStatePath(bootstrap.profileDir)
  }

  private async mutate(
    operation: (state: DesktopPluginEntryState) => DesktopPluginEntryState,
  ): Promise<void> {
    await ensurePrivateStateDirectory(this.statePath)
    await withFileLock(this.statePath, async () => {
      const next = parseState(operation(readDesktopPluginEntryState(this.bootstrap.profileDir)))
      await writeFileAtomic(this.statePath, renderState(next), {
        mode: STATE_FILE_MODE,
        dirMode: STATE_DIRECTORY_MODE,
      })
    })
  }

  private currentEntry(entryId: string) {
    if (!ENTRY_ID_PATTERN.test(entryId)) return undefined
    return [...this.bootstrap.loader.entries()]
      .find(entry => !entry.options.group && entry.id === entryId)
  }

  snapshot(): DesktopPluginEntrySnapshot {
    const state = readDesktopPluginEntryState(this.bootstrap.profileDir)
    const stored = new Map(state.entries.map(entry => [entry.id, entry]))
    const categoryIds = new Set([
      DEEPSEEK_PLUGIN_CATEGORY_ID,
      CODEX_PLUGIN_CATEGORY_ID,
      ...state.categories.map(category => category.id),
    ])
    const entries: DesktopPluginEntryView[] = []
    for (const entry of this.bootstrap.loader.entries()) {
      if (entry.options.group) continue
      const saved = stored.get(entry.id)
      const runtimeEnabled = !entry.disabled
      const enabled = saved?.enabled ?? runtimeEnabled
      entries.push(Object.freeze({
        entryId: entry.id,
        moduleName: entry.options.name,
        enabled,
        runtimeEnabled,
        categoryId: saved?.category !== undefined && categoryIds.has(saved.category)
          ? saved.category
          : defaultCategory(entry.id),
      }))
    }
    const categories: DesktopPluginCategoryView[] = [
      Object.freeze({ id: DEEPSEEK_PLUGIN_CATEGORY_ID, name: 'DeepSeek Harness', builtIn: true }),
      Object.freeze({ id: CODEX_PLUGIN_CATEGORY_ID, name: 'Codex Harness', builtIn: true }),
      ...state.categories.map(category => Object.freeze({ ...category, builtIn: false })),
    ]
    return Object.freeze({
      entries: Object.freeze(entries),
      categories: Object.freeze(categories),
      restartRequired: entries.some(entry => entry.enabled !== entry.runtimeEnabled),
    })
  }

  async setEnabled(entryId: string, enabled: boolean): Promise<void> {
    if (this.currentEntry(entryId) === undefined) {
      throw new Error(`${BIN_NAME}: Loader plugin entry is unavailable`)
    }
    await this.mutate(state => {
      const updates = new Map(state.entries.map(entry => [entry.id, { ...entry }]))
      const current = updates.get(entryId) ?? { id: entryId }
      updates.set(entryId, { ...current, enabled })
      if (enabled && entryId === CODEX_HARNESS_ENTRY_ID) {
        const peer = updates.get(DEEPSEEK_HARNESS_ENTRY_ID) ?? { id: DEEPSEEK_HARNESS_ENTRY_ID }
        updates.set(DEEPSEEK_HARNESS_ENTRY_ID, { ...peer, enabled: false })
      } else if (enabled && entryId === DEEPSEEK_HARNESS_ENTRY_ID) {
        const peer = updates.get(CODEX_HARNESS_ENTRY_ID) ?? { id: CODEX_HARNESS_ENTRY_ID }
        updates.set(CODEX_HARNESS_ENTRY_ID, { ...peer, enabled: false })
      }
      return { ...state, entries: [...updates.values()] }
    })
  }

  async createCategory(name: string): Promise<string> {
    const normalized = parseCategoryName(name)
    const id = `custom_${randomBytes(16).toString('hex')}`
    await this.mutate(state => {
      if (state.categories.some(category => category.name.toLocaleLowerCase() === normalized.toLocaleLowerCase())) {
        throw new Error(`${BIN_NAME}: plugin category already exists`)
      }
      if (state.categories.length >= MAX_CATEGORIES) {
        throw new Error(`${BIN_NAME}: too many plugin categories`)
      }
      return { ...state, categories: [...state.categories, { id, name: normalized }] }
    })
    return id
  }

  async assignCategory(entryId: string, categoryId: string): Promise<void> {
    if (this.currentEntry(entryId) === undefined) {
      throw new Error(`${BIN_NAME}: Loader plugin entry is unavailable`)
    }
    await this.mutate(state => {
      const custom = state.categories.some(category => category.id === categoryId)
      if (!custom && categoryId !== DEEPSEEK_PLUGIN_CATEGORY_ID && categoryId !== CODEX_PLUGIN_CATEGORY_ID) {
        throw new Error(`${BIN_NAME}: plugin category is unavailable`)
      }
      const updates = new Map(state.entries.map(entry => [entry.id, { ...entry }]))
      const current = updates.get(entryId) ?? { id: entryId }
      if (categoryId === defaultCategory(entryId)) {
        const { category: _category, ...rest } = current
        if (rest.enabled === undefined) updates.delete(entryId)
        else updates.set(entryId, rest)
      } else {
        updates.set(entryId, { ...current, category: categoryId })
      }
      return { ...state, entries: [...updates.values()] }
    })
  }

  async deleteCategory(categoryId: string): Promise<void> {
    if (!CUSTOM_CATEGORY_ID_PATTERN.test(categoryId)) {
      throw new Error(`${BIN_NAME}: built-in plugin categories cannot be deleted`)
    }
    await this.mutate(state => {
      if (!state.categories.some(category => category.id === categoryId)) {
        throw new Error(`${BIN_NAME}: plugin category is unavailable`)
      }
      const entries = state.entries.flatMap(entry => {
        if (entry.category !== categoryId) return [entry]
        const { category: _category, ...rest } = entry
        return rest.enabled === undefined ? [] : [rest]
      })
      return {
        ...state,
        categories: state.categories.filter(category => category.id !== categoryId),
        entries,
      }
    })
  }

  /** Apply persisted nested-tree overrides after Includes have mounted. */
  async reconcile(): Promise<void> {
    const overrides = readDesktopPluginEntryOverrides(this.bootstrap.profileDir)
    for (const [entryId, enabled] of overrides) {
      const entry = this.currentEntry(entryId)
      if (entry === undefined || !entryId.includes(':')) continue
      if (!entry.disabled === enabled) continue
      // The Profile state is the persistence owner. Calling Loader.update()
      // here would write into the Include's source file (which may be a
      // pinned upstream preset); Entry.update() gives the same Cordis HMR
      // lifecycle without modifying that owned source.
      await entry.update({ disabled: !enabled })
    }
  }
}

export const desktopPluginEntryLimits = Object.freeze({
  maxEntries: MAX_ENTRIES,
  maxCategories: MAX_CATEGORIES,
  maxCategoryNameLength: MAX_CATEGORY_NAME_LENGTH,
  entryIdPattern: ENTRY_ID_PATTERN,
  customCategoryIdPattern: CUSTOM_CATEGORY_ID_PATTERN,
})
