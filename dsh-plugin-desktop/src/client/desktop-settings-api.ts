/** Same-origin browser client for launcher-owned Desktop settings operations. */

const SETTINGS_PATH = '/api/desktop/settings'
const PROFILE_CREATE_PATH = '/api/desktop/profiles/create'
const PROFILE_SELECT_PATH = '/api/desktop/profiles/select'
const PROFILE_DELETE_PATH = '/api/desktop/profiles/delete'
const MARKET_SELECT_PATH = '/api/desktop/market/select'
const PLUGINS_PATH = '/api/desktop/plugins'
const PLUGIN_TOGGLE_PATH = '/api/desktop/plugins/toggle'
const PLUGIN_ENTRY_TOGGLE_PATH = '/api/desktop/plugin-entries/toggle'
const HARNESS_CREATE_PATH = '/api/desktop/harnesses/create'
const HARNESS_ASSIGN_PATH = '/api/desktop/harnesses/assign'
const HARNESS_DELETE_PATH = '/api/desktop/harnesses/delete'
const PLUGIN_RESTART_PATH = '/api/desktop/plugins/restart'
const TERMINAL_OPEN_PATH = '/api/desktop/terminal/open'
const MAX_PROFILES = 256
const MAX_PROFILE_NAME_LENGTH = 255
const MAX_PLUGIN_BUNDLES = 1024
const BUNDLE_ID_PATTERN = /^bundle_[A-Za-z0-9_-]{32}$/u
const PACKAGE_NAME_PATTERN = /^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/u
const ENTRY_ID_PATTERN = /^[A-Za-z0-9._:@/-]{1,256}$/u
const HARNESS_ID_PATTERN = /^(?:deepseek|codex|common|custom_[a-f0-9]{32})$/u
const MAX_PLUGIN_ENTRIES = 2048
const MAX_PLUGIN_HARNESSES = 67

/** Launcher-supported plugin market implementations. */
export type DesktopMarketProvider = 'disabled' | 'community-market' | 'dsh-market'

/** Safe profile projection returned to the renderer. */
export interface DesktopProfileView {
  readonly name: string
  readonly exists: boolean
  readonly webCapable: boolean
  readonly selectable: boolean
  readonly deletable: boolean
}

/** Market selection fixed for the running generation. */
export interface DesktopMarketView {
  readonly requested: DesktopMarketProvider
  readonly effective: DesktopMarketProvider
  readonly legacyDefaulted: boolean
}

/** Complete launcher-owned settings projection. */
export interface DesktopSettingsView {
  readonly current: string
  readonly profiles: readonly DesktopProfileView[]
  readonly market: DesktopMarketView
}

/** One direct Profile bundle that can be displayed and, when mutable, toggled. */
export interface DesktopPluginBundleView {
  readonly bundleId: string
  readonly packageName: string
  readonly status: 'active' | 'disabled'
  readonly mutable: boolean
}

export interface DesktopPluginEntryView {
  readonly entryId: string
  readonly moduleName: string
  readonly enabled: boolean
  readonly harnessId: string
  readonly engine: boolean
  readonly locked: boolean
}

export interface DesktopPluginHarnessView {
  readonly id: string
  readonly name: string
  readonly builtIn: boolean
  readonly engine?: string
  readonly selectable: boolean
}

/** Fresh direct-bundle state plus whether the running generation is stale. */
export interface DesktopPluginsView {
  readonly bundles: readonly DesktopPluginBundleView[]
  readonly entries: readonly DesktopPluginEntryView[]
  readonly harnesses: readonly DesktopPluginHarnessView[]
  readonly primaryHarness: string
  readonly restartRequired: boolean
}

/** A persisted selection that requires a new Desktop generation. */
export interface DesktopRestartAcceptance {
  readonly accepted: true
  readonly restartRequired: boolean
}

/** Browser operations consumed by the Desktop settings section. */
export interface DesktopSettingsApi {
  read(): Promise<DesktopSettingsView>
  createProfile(name: string): Promise<DesktopSettingsView>
  selectProfile(name: string): Promise<DesktopRestartAcceptance>
  deleteProfile(name: string): Promise<DesktopSettingsView>
  selectMarket(provider: DesktopMarketProvider): Promise<DesktopRestartAcceptance>
  readPlugins(): Promise<DesktopPluginsView>
  setPluginEnabled(bundleId: string, enabled: boolean): Promise<DesktopPluginsView>
  setPluginEntryEnabled(entryId: string, enabled: boolean): Promise<DesktopPluginsView>
  createHarness(name: string, engine?: string): Promise<DesktopPluginsView>
  assignHarness(entryId: string, harnessId: string): Promise<DesktopPluginsView>
  deleteHarness(harnessId: string): Promise<DesktopPluginsView>
  restartPlugins(): Promise<DesktopRestartAcceptance>
  openTerminal(): Promise<void>
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isMarketProvider(value: unknown): value is DesktopMarketProvider {
  return value === 'disabled' || value === 'community-market' || value === 'dsh-market'
}

function parseProfile(value: unknown): DesktopProfileView {
  if (!isObject(value)
    || typeof value.name !== 'string'
    || value.name.length === 0
    || value.name.length > MAX_PROFILE_NAME_LENGTH
    || typeof value.exists !== 'boolean'
    || typeof value.webCapable !== 'boolean'
    || typeof value.selectable !== 'boolean'
    || typeof value.deletable !== 'boolean') {
    throw new Error('dsh-plugin-desktop: invalid profile settings response')
  }
  return Object.freeze({
    name: value.name,
    exists: value.exists,
    webCapable: value.webCapable,
    selectable: value.selectable,
    deletable: value.deletable,
  })
}

/** Validate the bounded settings projection before it reaches React state. */
export function parseDesktopSettingsView(value: unknown): DesktopSettingsView {
  if (!isObject(value)
    || typeof value.current !== 'string'
    || value.current.length === 0
    || value.current.length > MAX_PROFILE_NAME_LENGTH
    || !Array.isArray(value.profiles)
    || value.profiles.length > MAX_PROFILES
    || !isObject(value.market)
    || !isMarketProvider(value.market.requested)
    || !isMarketProvider(value.market.effective)
    || typeof value.market.legacyDefaulted !== 'boolean') {
    throw new Error('dsh-plugin-desktop: invalid Desktop settings response')
  }
  const profiles = value.profiles.map(parseProfile)
  if (new Set(profiles.map(profile => profile.name)).size !== profiles.length) {
    throw new Error('dsh-plugin-desktop: duplicate profile in settings response')
  }
  return Object.freeze({
    current: value.current,
    profiles: Object.freeze(profiles),
    market: Object.freeze({
      requested: value.market.requested,
      effective: value.market.effective,
      legacyDefaulted: value.market.legacyDefaulted,
    }),
  })
}

/** Validate the bounded plugin projection before it reaches React state. */
export function parseDesktopPluginsView(value: unknown): DesktopPluginsView {
  if (!isObject(value)
    || !Array.isArray(value.bundles)
    || value.bundles.length > MAX_PLUGIN_BUNDLES
    || (value.entries !== undefined && (!Array.isArray(value.entries) || value.entries.length > MAX_PLUGIN_ENTRIES))
    || (value.harnesses !== undefined && (!Array.isArray(value.harnesses) || value.harnesses.length > MAX_PLUGIN_HARNESSES))
    || typeof value.restartRequired !== 'boolean'
    || typeof value.primaryHarness !== 'string'
    || !HARNESS_ID_PATTERN.test(value.primaryHarness)) {
    throw new Error('dsh-plugin-desktop: invalid Desktop plugins response')
  }
  const bundles = value.bundles.map((raw): DesktopPluginBundleView => {
    if (!isObject(raw)
      || typeof raw.bundleId !== 'string'
      || !BUNDLE_ID_PATTERN.test(raw.bundleId)
      || typeof raw.packageName !== 'string'
      || raw.packageName.length > 214
      || !PACKAGE_NAME_PATTERN.test(raw.packageName)
      || (raw.status !== 'active' && raw.status !== 'disabled')
      || typeof raw.mutable !== 'boolean') {
      throw new Error('dsh-plugin-desktop: invalid Desktop plugin bundle response')
    }
    return Object.freeze({
      bundleId: raw.bundleId,
      packageName: raw.packageName,
      status: raw.status,
      mutable: raw.mutable,
    })
  })
  if (new Set(bundles.map(bundle => bundle.bundleId)).size !== bundles.length
    || new Set(bundles.map(bundle => bundle.packageName)).size !== bundles.length) {
    throw new Error('dsh-plugin-desktop: duplicate Desktop plugin bundle')
  }
  const rawEntries = value.entries ?? []
  const entries = (rawEntries as unknown[]).map((raw): DesktopPluginEntryView => {
    if (!isObject(raw)
      || typeof raw.entryId !== 'string'
      || !ENTRY_ID_PATTERN.test(raw.entryId)
      || typeof raw.moduleName !== 'string'
      || raw.moduleName.length === 0
      || raw.moduleName.length > 512
      || typeof raw.enabled !== 'boolean'
      || typeof raw.harnessId !== 'string'
      || !HARNESS_ID_PATTERN.test(raw.harnessId)
      || typeof raw.engine !== 'boolean'
      || typeof raw.locked !== 'boolean') {
      throw new Error('dsh-plugin-desktop: invalid Desktop plugin entry response')
    }
    return Object.freeze({
      entryId: raw.entryId,
      moduleName: raw.moduleName,
      enabled: raw.enabled,
      harnessId: raw.harnessId,
      engine: raw.engine,
      locked: raw.locked,
    })
  })
  const rawHarnesses = value.harnesses ?? [
    { id: 'deepseek', name: 'DeepSeek Harness', builtIn: true, engine: 'agent-loop', selectable: true },
    { id: 'codex', name: 'Codex Harness', builtIn: true, engine: 'codex-harness', selectable: true },
    { id: 'common', name: 'Common Plugins', builtIn: true, selectable: false },
  ]
  const harnesses = (rawHarnesses as unknown[]).map((raw): DesktopPluginHarnessView => {
    if (!isObject(raw)
      || typeof raw.id !== 'string'
      || !HARNESS_ID_PATTERN.test(raw.id)
      || typeof raw.name !== 'string'
      || raw.name.trim().length === 0
      || raw.name.length > 64
      || typeof raw.builtIn !== 'boolean'
      || (raw.engine !== undefined
        && (typeof raw.engine !== 'string' || !ENTRY_ID_PATTERN.test(raw.engine)))
      || typeof raw.selectable !== 'boolean') {
      throw new Error('dsh-plugin-desktop: invalid Desktop plugin harness response')
    }
    return Object.freeze({
      id: raw.id,
      name: raw.name,
      builtIn: raw.builtIn,
      ...(raw.engine === undefined ? {} : { engine: raw.engine }),
      selectable: raw.selectable,
    })
  })
  if (new Set(entries.map(entry => entry.entryId)).size !== entries.length
    || new Set(harnesses.map(harness => harness.id)).size !== harnesses.length
    || entries.some(entry => !harnesses.some(harness => harness.id === entry.harnessId))) {
    throw new Error('dsh-plugin-desktop: duplicate or missing Desktop plugin harness')
  }
  return Object.freeze({
    bundles: Object.freeze(bundles),
    entries: Object.freeze(entries),
    harnesses: Object.freeze(harnesses),
    primaryHarness: value.primaryHarness,
    restartRequired: value.restartRequired,
  })
}

/** Validate restart acknowledgement returned before the Host generation exits. */
export function parseDesktopRestartAcceptance(value: unknown): DesktopRestartAcceptance {
  if (!isObject(value) || value.accepted !== true || typeof value.restartRequired !== 'boolean') {
    throw new Error('dsh-plugin-desktop: invalid Desktop restart response')
  }
  return Object.freeze({ accepted: true, restartRequired: value.restartRequired })
}

/** Validate the exact acknowledgement returned by a Desktop side effect. */
export function parseDesktopActionAcceptance(value: unknown): void {
  if (!isObject(value)
    || Object.keys(value).length !== 1
    || value.accepted !== true) {
    throw new Error('dsh-plugin-desktop: invalid Desktop action response')
  }
}

async function readResponse(response: Response): Promise<unknown> {
  if (!response.ok) {
    throw new Error(`dsh-plugin-desktop: Desktop settings request failed (${String(response.status)})`)
  }
  try {
    return await response.json() as unknown
  } catch {
    throw new Error('dsh-plugin-desktop: Desktop settings response was not JSON')
  }
}

function post(fetcher: FetchLike, path: string, body: object): Promise<Response> {
  return fetcher(path, {
    method: 'POST',
    credentials: 'same-origin',
    redirect: 'error',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
}

/** Construct the default same-origin API, with a fetch seam for focused tests. */
export function createDesktopSettingsApi(fetcher: FetchLike = globalThis.fetch.bind(globalThis)): DesktopSettingsApi {
  return Object.freeze({
    async read() {
      const response = await fetcher(SETTINGS_PATH, {
        method: 'GET',
        credentials: 'same-origin',
        redirect: 'error',
        cache: 'no-store',
        headers: { 'Accept': 'application/json' },
      })
      return parseDesktopSettingsView(await readResponse(response))
    },
    async createProfile(name: string) {
      return parseDesktopSettingsView(await readResponse(await post(fetcher, PROFILE_CREATE_PATH, { name })))
    },
    async selectProfile(name: string) {
      return parseDesktopRestartAcceptance(await readResponse(await post(fetcher, PROFILE_SELECT_PATH, { name })))
    },
    async deleteProfile(name: string) {
      return parseDesktopSettingsView(await readResponse(await post(fetcher, PROFILE_DELETE_PATH, { name })))
    },
    async selectMarket(provider: DesktopMarketProvider) {
      return parseDesktopRestartAcceptance(await readResponse(await post(fetcher, MARKET_SELECT_PATH, { provider })))
    },
    async readPlugins() {
      const response = await fetcher(PLUGINS_PATH, {
        method: 'GET',
        credentials: 'same-origin',
        redirect: 'error',
        cache: 'no-store',
        headers: { 'Accept': 'application/json' },
      })
      return parseDesktopPluginsView(await readResponse(response))
    },
    async setPluginEnabled(bundleId: string, enabled: boolean) {
      const value = await readResponse(await post(fetcher, PLUGIN_TOGGLE_PATH, { bundleId, enabled }))
      if (!isObject(value) || value.accepted !== true) {
        throw new Error('dsh-plugin-desktop: invalid Desktop plugin toggle response')
      }
      return parseDesktopPluginsView(value)
    },
    async setPluginEntryEnabled(entryId: string, enabled: boolean) {
      const value = await readResponse(await post(fetcher, PLUGIN_ENTRY_TOGGLE_PATH, { entryId, enabled }))
      if (!isObject(value) || value.accepted !== true) {
        throw new Error('dsh-plugin-desktop: invalid Desktop plugin entry toggle response')
      }
      return parseDesktopPluginsView(value)
    },
    async createHarness(name: string, engine?: string) {
      const value = await readResponse(await post(fetcher, HARNESS_CREATE_PATH, { name, ...(engine === undefined ? {} : { engine }) }))
      if (!isObject(value) || value.accepted !== true) {
        throw new Error('dsh-plugin-desktop: invalid Desktop harness response')
      }
      return parseDesktopPluginsView(value)
    },
    async assignHarness(entryId: string, harnessId: string) {
      const value = await readResponse(await post(fetcher, HARNESS_ASSIGN_PATH, { entryId, harnessId }))
      if (!isObject(value) || value.accepted !== true) {
        throw new Error('dsh-plugin-desktop: invalid Desktop harness response')
      }
      return parseDesktopPluginsView(value)
    },
    async deleteHarness(harnessId: string) {
      const value = await readResponse(await post(fetcher, HARNESS_DELETE_PATH, { harnessId }))
      if (!isObject(value) || value.accepted !== true) {
        throw new Error('dsh-plugin-desktop: invalid Desktop harness response')
      }
      return parseDesktopPluginsView(value)
    },
    async restartPlugins() {
      return parseDesktopRestartAcceptance(await readResponse(await post(fetcher, PLUGIN_RESTART_PATH, {})))
    },
    async openTerminal() {
      parseDesktopActionAcceptance(await readResponse(await post(fetcher, TERMINAL_OPEN_PATH, {})))
    },
  })
}

export const desktopSettingsPaths = Object.freeze({
  settings: SETTINGS_PATH,
  profileCreate: PROFILE_CREATE_PATH,
  profileSelect: PROFILE_SELECT_PATH,
  profileDelete: PROFILE_DELETE_PATH,
  marketSelect: MARKET_SELECT_PATH,
  plugins: PLUGINS_PATH,
  pluginToggle: PLUGIN_TOGGLE_PATH,
  pluginEntryToggle: PLUGIN_ENTRY_TOGGLE_PATH,
  harnessCreate: HARNESS_CREATE_PATH,
  harnessAssign: HARNESS_ASSIGN_PATH,
  harnessDelete: HARNESS_DELETE_PATH,
  pluginRestart: PLUGIN_RESTART_PATH,
  terminalOpen: TERMINAL_OPEN_PATH,
})
