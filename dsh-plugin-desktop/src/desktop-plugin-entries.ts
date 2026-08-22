/** Profile-scoped harness plugin sets over the effective Cordis Loader plugin tree. */

import { randomBytes } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { chmod, lstat, mkdir } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'
import type { Loader } from '@deepseek-ai/cordis-plugin-loader'
import { withFileLock, writeFileAtomic } from '@deepseek-ai/dsh-atomic-write'
import {
  COMMON_HARNESS_ID,
  CUSTOM_HARNESS_ID_PATTERN,
  DEEPSEEK_HARNESS_ID,
  HARNESS_PROVIDERS,
  harnessProviderForEngine,
  type CustomHarnessId,
  type HarnessId,
  type PrimaryHarnessId,
} from './harnesses.ts'

const BIN_NAME = 'dsh-plugin-desktop'
const STATE_VERSION = 2
const STATE_FILE = join('.rundeep', 'plugins.json')
const STATE_FILE_MODE = 0o600
const STATE_DIRECTORY_MODE = 0o700
const MAX_STATE_BYTES = 256 * 1024
const MAX_ENTRIES = 2048
const MAX_HARNESSES = 64
const MAX_HARNESS_NAME_LENGTH = 64
const ENTRY_ID_PATTERN = /^[A-Za-z0-9._:@/-]{1,256}$/u
const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/u

/** Stable built-in group names; the client localizes built-in ids. */
export const BUILTIN_HARNESS_NAMES: Readonly<Record<string, string>> = Object.freeze({
  [DEEPSEEK_HARNESS_ID]: 'DeepSeek Harness',
  codex: 'Codex Harness',
  [COMMON_HARNESS_ID]: 'Common Plugins',
})

/** Legacy v1 category ids that map to built-in harnesses during migration. */
const LEGACY_BUILTIN_CATEGORY_IDS = new Set<string>([DEEPSEEK_HARNESS_ID, 'codex'])

interface StoredPluginHarness {
  readonly id: CustomHarnessId
  readonly name: string
  readonly engine?: string
}

interface StoredPluginEntry {
  readonly id: string
  readonly enabled?: boolean
  readonly harness?: HarnessId
}

export interface DesktopPluginEntryState {
  readonly version: 2
  readonly harnesses: readonly StoredPluginHarness[]
  readonly entries: readonly StoredPluginEntry[]
}

export interface DesktopPluginEntryView {
  readonly entryId: string
  readonly moduleName: string
  /** Persisted state used by the next generation after harness linking. */
  readonly enabled: boolean
  /** Effective state in the currently running Loader tree. */
  readonly runtimeEnabled: boolean
  /** Harness or generic group owning this entry. */
  readonly harnessId: HarnessId
  /** Whether this Loader row is a harness AgentFactory. */
  readonly engine: boolean
  /** Whether the primary harness selection, not the switch, owns this state. */
  readonly locked: boolean
}

export interface DesktopPluginHarnessView {
  readonly id: HarnessId
  readonly name: string
  readonly builtIn: boolean
  /** Loader row id of the AgentFactory owned by this harness; absent for common. */
  readonly engine?: string
  /** Whether this harness can own the primary AgentFactory selection. */
  readonly selectable: boolean
}

export interface DesktopPluginEntrySnapshot {
  readonly entries: readonly DesktopPluginEntryView[]
  readonly harnesses: readonly DesktopPluginHarnessView[]
  /** Primary AgentFactory id fixed for the running generation. */
  readonly primaryHarness: PrimaryHarnessId
  readonly restartRequired: boolean
}

export interface DesktopPluginEntriesBootstrap {
  readonly profileDir: string
  readonly loader: Pick<Loader, 'entries'>
  /** Primary AgentFactory selected before the current generation mounted. */
  readonly primaryHarness: PrimaryHarnessId
}

function emptyState(): DesktopPluginEntryState {
  return { version: STATE_VERSION, harnesses: [], entries: [] }
}

function isExactKeys(value: Record<string, unknown>, allowed: readonly string[]): boolean {
  const keys = Object.keys(value)
  return keys.every(key => allowed.includes(key))
}

function parseHarnessName(value: unknown): string {
  if (typeof value !== 'string') throw new Error('harness name must be a string')
  const name = value.trim()
  if (name.length === 0 || name.length > MAX_HARNESS_NAME_LENGTH
    || CONTROL_CHARACTER_PATTERN.test(name)) {
    throw new Error('harness name is invalid')
  }
  return name
}

function parseLegacyV1State(value: unknown): DesktopPluginEntryState {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('state must be an object')
  }
  const record = value as Record<string, unknown>
  if (!isExactKeys(record, ['version', 'entries', 'categories'])
    || record.version !== 1
    || !Array.isArray(record.entries)
    || !Array.isArray(record.categories)
    || record.entries.length > MAX_ENTRIES
    || record.categories.length > MAX_HARNESSES) {
    throw new Error('state shape is invalid')
  }

  const harnessNames = new Set<string>(Object.values(BUILTIN_HARNESS_NAMES)
    .map(name => name.toLocaleLowerCase()))
  const harnesses: StoredPluginHarness[] = []
  const harnessIds = new Set<string>()
  for (const raw of record.categories) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error('category is invalid')
    }
    const category = raw as Record<string, unknown>
    if (!isExactKeys(category, ['id', 'name'])
      || typeof category.id !== 'string'
      || !CUSTOM_HARNESS_ID_PATTERN.test(category.id)) {
      throw new Error('category is invalid')
    }
    const name = parseHarnessName(category.name)
    const normalizedName = name.toLocaleLowerCase()
    if (harnessIds.has(category.id) || harnessNames.has(normalizedName)) {
      throw new Error('category is duplicated')
    }
    harnessIds.add(category.id)
    harnessNames.add(normalizedName)
    harnesses.push({ id: category.id as CustomHarnessId, name })
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
          || (!harnessIds.has(entry.category)
            && !LEGACY_BUILTIN_CATEGORY_IDS.has(entry.category))))
      || (entry.enabled === undefined && entry.category === undefined)
      || entryIds.has(entry.id)) {
      throw new Error('entry state is invalid')
    }
    entryIds.add(entry.id)
    entries.push({
      id: entry.id,
      ...(entry.enabled === undefined ? {} : { enabled: entry.enabled }),
      ...(entry.category === undefined ? {} : { harness: entry.category as HarnessId }),
    })
  }
  return { version: STATE_VERSION, harnesses, entries }
}

function parseState(value: unknown): DesktopPluginEntryState {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('state must be an object')
  }
  const record = value as Record<string, unknown>
  if (record.version === 1) return parseLegacyV1State(value)
  if (!isExactKeys(record, ['version', 'harnesses', 'entries'])
    || record.version !== STATE_VERSION
    || !Array.isArray(record.harnesses)
    || !Array.isArray(record.entries)
    || record.harnesses.length > MAX_HARNESSES
    || record.entries.length > MAX_ENTRIES) {
    throw new Error('state shape is invalid')
  }

  const harnessNames = new Set<string>(Object.values(BUILTIN_HARNESS_NAMES)
    .map(name => name.toLocaleLowerCase()))
  const harnesses: StoredPluginHarness[] = []
  const harnessIds = new Set<string>()
  const engineOwners = new Map<string, CustomHarnessId>()
  for (const raw of record.harnesses) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error('harness is invalid')
    }
    const harness = raw as Record<string, unknown>
    if (!isExactKeys(harness, ['id', 'name', 'engine'])
      || typeof harness.id !== 'string'
      || !CUSTOM_HARNESS_ID_PATTERN.test(harness.id)
      || (harness.engine !== undefined
        && (typeof harness.engine !== 'string' || !ENTRY_ID_PATTERN.test(harness.engine)))) {
      throw new Error('harness is invalid')
    }
    const name = parseHarnessName(harness.name)
    const normalizedName = name.toLocaleLowerCase()
    if (harnessIds.has(harness.id) || harnessNames.has(normalizedName)) {
      throw new Error('harness is duplicated')
    }
    harnessIds.add(harness.id)
    harnessNames.add(normalizedName)
    if (harness.engine !== undefined) {
      const engine = harness.engine as string
      if (engineOwners.has(engine)) {
        throw new Error('harness engine is duplicated')
      }
      engineOwners.set(engine, harness.id as CustomHarnessId)
    }
    harnesses.push({
      id: harness.id as CustomHarnessId,
      name,
      ...(harness.engine === undefined ? {} : { engine: harness.engine as string }),
    })
  }

  const entries: StoredPluginEntry[] = []
  const entryIds = new Set<string>()
  for (const raw of record.entries) {
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new Error('entry state is invalid')
    }
    const entry = raw as Record<string, unknown>
    if (!isExactKeys(entry, ['id', 'enabled', 'harness'])
      || typeof entry.id !== 'string'
      || !ENTRY_ID_PATTERN.test(entry.id)
      || (entry.enabled !== undefined && typeof entry.enabled !== 'boolean')
      || (entry.harness !== undefined
        && (typeof entry.harness !== 'string'
          || (!harnessIds.has(entry.harness)
            && entry.harness !== DEEPSEEK_HARNESS_ID
            && entry.harness !== 'codex'
            && entry.harness !== COMMON_HARNESS_ID)))
      || (entry.enabled === undefined && entry.harness === undefined)
      || entryIds.has(entry.id)) {
      throw new Error('entry state is invalid')
    }
    entryIds.add(entry.id)
    entries.push({
      id: entry.id,
      ...(entry.enabled === undefined ? {} : { enabled: entry.enabled }),
      ...(entry.harness === undefined ? {} : { harness: entry.harness as HarnessId }),
    })
  }
  return { version: STATE_VERSION, harnesses, entries }
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

/** Return the harness or group that owns one Loader row by default. */
export function defaultHarnessId(entryId: string): HarnessId {
  return harnessProviderForEngine(entryId)?.id ?? COMMON_HARNESS_ID
}

/** Fall back to DeepSeek when a stored primary harness id is no longer valid. */
export function resolvePrimaryHarness(
  value: PrimaryHarnessId,
  state: DesktopPluginEntryState,
): PrimaryHarnessId {
  if (value === DEEPSEEK_HARNESS_ID || value === 'codex') return value
  return state.harnesses.some(harness => harness.id === value) ? value : DEEPSEEK_HARNESS_ID
}

/** Every engine row id owned by a built-in or stored custom harness. */
function engineRows(state: DesktopPluginEntryState): ReadonlyMap<string, HarnessId> {
  const owners = new Map<string, HarnessId>()
  for (const provider of HARNESS_PROVIDERS) owners.set(provider.engine, provider.id)
  for (const harness of state.harnesses) {
    if (harness.engine !== undefined) owners.set(harness.engine, harness.id)
  }
  return owners
}

/**
 * Compute the intent patches that link the primary harness selection to every
 * Loader row: the primary engine and its plugin set load, every other harness
 * set stays disabled, and the common group keeps manual state.
 */
export function desktopHarnessEntryOverrides(
  profileDir: string,
  primaryHarness: PrimaryHarnessId,
): ReadonlyMap<string, boolean> {
  const state = readDesktopPluginEntryState(profileDir)
  const primary = resolvePrimaryHarness(primaryHarness, state)
  const engines = engineRows(state)
  const overrides = new Map<string, boolean>()
  for (const [engine, harnessId] of engines) {
    overrides.set(engine, harnessId === primary)
  }
  for (const entry of state.entries) {
    if (engines.has(entry.id)) continue
    const harnessId = entry.harness ?? COMMON_HARNESS_ID
    if (harnessId === primary) {
      if (entry.enabled === false) overrides.set(entry.id, false)
    } else if (harnessId === COMMON_HARNESS_ID) {
      if (entry.enabled !== undefined) overrides.set(entry.id, entry.enabled)
    } else {
      overrides.set(entry.id, false)
    }
  }
  return overrides
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

/** Persistent harness-set and enablement mutations over the effective Loader tree. */
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

  private isEngineRow(state: DesktopPluginEntryState, entryId: string): boolean {
    return engineRows(state).has(entryId)
  }

  snapshot(): DesktopPluginEntrySnapshot {
    const state = readDesktopPluginEntryState(this.bootstrap.profileDir)
    const stored = new Map(state.entries.map(entry => [entry.id, entry]))
    const engines = engineRows(state)
    const primary = resolvePrimaryHarness(this.bootstrap.primaryHarness, state)
    const harnesses: DesktopPluginHarnessView[] = [
      ...HARNESS_PROVIDERS.map(provider => Object.freeze({
        id: provider.id,
        name: provider.name,
        builtIn: true,
        engine: provider.engine,
        selectable: true,
      })),
      Object.freeze({
        id: COMMON_HARNESS_ID as HarnessId,
        name: BUILTIN_HARNESS_NAMES[COMMON_HARNESS_ID] ?? 'Common Plugins',
        builtIn: true,
        selectable: false,
      }),
      ...state.harnesses.map(harness => Object.freeze({
        id: harness.id,
        name: harness.name,
        builtIn: false,
        ...(harness.engine === undefined ? {} : { engine: harness.engine }),
        selectable: harness.engine !== undefined,
      })),
    ]
    const entries: DesktopPluginEntryView[] = []
    for (const entry of this.bootstrap.loader.entries()) {
      if (entry.options.group) continue
      const saved = stored.get(entry.id)
      const runtimeEnabled = !entry.disabled
      const engineHarness = engines.get(entry.id)
      let harnessId: HarnessId
      let enabled: boolean
      let locked: boolean
      if (engineHarness !== undefined) {
        harnessId = engineHarness
        enabled = engineHarness === primary
        locked = true
      } else {
        harnessId = saved?.harness ?? COMMON_HARNESS_ID
        if (harnessId === primary) {
          enabled = saved?.enabled ?? true
          locked = false
        } else if (harnessId === COMMON_HARNESS_ID) {
          enabled = saved?.enabled ?? runtimeEnabled
          locked = false
        } else {
          enabled = false
          locked = true
        }
      }
      entries.push(Object.freeze({
        entryId: entry.id,
        moduleName: entry.options.name,
        enabled,
        runtimeEnabled,
        harnessId,
        engine: engineHarness !== undefined,
        locked,
      }))
    }
    return Object.freeze({
      entries: Object.freeze(entries),
      harnesses: Object.freeze(harnesses),
      primaryHarness: primary,
      restartRequired: entries.some(entry => entry.enabled !== entry.runtimeEnabled),
    })
  }

  async setEnabled(entryId: string, enabled: boolean): Promise<void> {
    if (this.currentEntry(entryId) === undefined) {
      throw new Error(`${BIN_NAME}: Loader plugin entry is unavailable`)
    }
    await this.mutate(state => {
      if (this.isEngineRow(state, entryId)) {
        throw new Error(`${BIN_NAME}: harness engine entries are controlled by the primary harness setting`)
      }
      const updates = new Map(state.entries.map(entry => [entry.id, { ...entry }]))
      const current = updates.get(entryId) ?? { id: entryId }
      updates.set(entryId, { ...current, enabled })
      return { ...state, entries: [...updates.values()] }
    })
  }

  async createHarness(name: string, engine?: string): Promise<CustomHarnessId> {
    const normalized = parseHarnessName(name)
    const engineEntry = engine === undefined ? undefined : this.currentEntry(engine)
    if (engine !== undefined && engineEntry === undefined) {
      throw new Error(`${BIN_NAME}: harness engine Loader entry is unavailable`)
    }
    const id: CustomHarnessId = `custom_${randomBytes(16).toString('hex')}`
    await this.mutate(state => {
      const takenNames = new Set<string>(Object.values(BUILTIN_HARNESS_NAMES)
        .map(value => value.toLocaleLowerCase()))
      for (const harness of state.harnesses) takenNames.add(harness.name.toLocaleLowerCase())
      if (takenNames.has(normalized.toLocaleLowerCase())) {
        throw new Error(`${BIN_NAME}: harness already exists`)
      }
      if (state.harnesses.length >= MAX_HARNESSES) {
        throw new Error(`${BIN_NAME}: too many harnesses`)
      }
      if (engine !== undefined && engineRows(state).has(engine)) {
        throw new Error(`${BIN_NAME}: harness engine Loader row is already owned`)
      }
      return {
        ...state,
        harnesses: [...state.harnesses, {
          id,
          name: normalized,
          ...(engine === undefined ? {} : { engine }),
        }],
      }
    })
    return id
  }

  async assignHarness(entryId: string, harnessId: string): Promise<void> {
    if (this.currentEntry(entryId) === undefined) {
      throw new Error(`${BIN_NAME}: Loader plugin entry is unavailable`)
    }
    await this.mutate(state => {
      if (this.isEngineRow(state, entryId)) {
        throw new Error(`${BIN_NAME}: harness engine entries cannot be reassigned`)
      }
      const custom = state.harnesses.some(harness => harness.id === harnessId)
      if (!custom && harnessId !== DEEPSEEK_HARNESS_ID
        && harnessId !== 'codex' && harnessId !== COMMON_HARNESS_ID) {
        throw new Error(`${BIN_NAME}: harness is unavailable`)
      }
      const updates = new Map(state.entries.map(entry => [entry.id, { ...entry }]))
      const current = updates.get(entryId) ?? { id: entryId }
      if (harnessId === defaultHarnessId(entryId)) {
        const { harness: _harness, ...rest } = current
        if (rest.enabled === undefined) updates.delete(entryId)
        else updates.set(entryId, rest)
      } else {
        updates.set(entryId, { ...current, harness: harnessId as HarnessId })
      }
      return { ...state, entries: [...updates.values()] }
    })
  }

  async deleteHarness(harnessId: string): Promise<void> {
    if (!CUSTOM_HARNESS_ID_PATTERN.test(harnessId)) {
      throw new Error(`${BIN_NAME}: built-in harnesses cannot be deleted`)
    }
    await this.mutate(state => {
      const target = state.harnesses.find(harness => harness.id === harnessId)
      if (target === undefined) {
        throw new Error(`${BIN_NAME}: harness is unavailable`)
      }
      if (target.id === this.bootstrap.primaryHarness) {
        throw new Error(`${BIN_NAME}: the active primary harness cannot be deleted`)
      }
      const entries = state.entries.flatMap(entry => {
        if (entry.harness !== harnessId) return [entry]
        const { harness: _harness, ...rest } = entry
        return rest.enabled === undefined ? [] : [rest]
      })
      return {
        ...state,
        harnesses: state.harnesses.filter(harness => harness.id !== harnessId),
        entries,
      }
    })
  }

  /** Apply persisted nested-tree overrides after Includes have mounted. */
  async reconcile(): Promise<void> {
    const overrides = desktopHarnessEntryOverrides(
      this.bootstrap.profileDir,
      this.bootstrap.primaryHarness,
    )
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
  maxHarnesses: MAX_HARNESSES,
  maxHarnessNameLength: MAX_HARNESS_NAME_LENGTH,
  entryIdPattern: ENTRY_ID_PATTERN,
  customHarnessIdPattern: CUSTOM_HARNESS_ID_PATTERN,
})
