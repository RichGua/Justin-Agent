/** Private same-origin Desktop settings API shared with the bundled renderer. */

import type { DesktopMarketProvider } from './desktop-market.ts'

/** Read the current Desktop-owned settings state. */
export const DESKTOP_SETTINGS_PATH = '/api/desktop/settings'

/** Create one safe Web profile without selecting it. */
export const DESKTOP_PROFILE_CREATE_PATH = '/api/desktop/profiles/create'

/** Select one compatible profile for the next Desktop generation. */
export const DESKTOP_PROFILE_SELECT_PATH = '/api/desktop/profiles/select'

/** Delete one inactive, user-created Web Profile. */
export const DESKTOP_PROFILE_DELETE_PATH = '/api/desktop/profiles/delete'

/** Persist the Market provider selected for the next Desktop generation. */
export const DESKTOP_MARKET_SELECT_PATH = '/api/desktop/market/select'

/** Read the active Profile's direct plugin bundles and pending restart state. */
export const DESKTOP_PLUGINS_PATH = '/api/desktop/plugins'

/** Persist one reversible direct-bundle enablement change. */
export const DESKTOP_PLUGIN_TOGGLE_PATH = '/api/desktop/plugins/toggle'

/** Persist one Cordis Loader entry enablement override. */
export const DESKTOP_PLUGIN_ENTRY_TOGGLE_PATH = '/api/desktop/plugin-entries/toggle'

/** Create one Profile-local harness definition. */
export const DESKTOP_HARNESS_CREATE_PATH = '/api/desktop/harnesses/create'

/** Assign one Loader entry to a harness or the common group. */
export const DESKTOP_HARNESS_ASSIGN_PATH = '/api/desktop/harnesses/assign'

/** Delete one user-created harness and return its entries to the common group. */
export const DESKTOP_HARNESS_DELETE_PATH = '/api/desktop/harnesses/delete'

/** Apply pending plugin changes through an orderly Desktop restart. */
export const DESKTOP_PLUGIN_RESTART_PATH = '/api/desktop/plugins/restart'

/** Open the launcher-owned DSH terminal without accepting command text. */
export const DESKTOP_TERMINAL_OPEN_PATH = '/api/desktop/terminal/open'

/** Export one local diagnostic archive through the launcher-owned flow. */
export const DESKTOP_DIAGNOSTICS_EXPORT_PATH = '/api/desktop/diagnostics/export'

/** Open the isolated native Profile creator without accepting a path. */
export const DESKTOP_PROFILE_CREATE_WINDOW_PATH = '/api/desktop/profiles/create-window'

/** Restore the last successful Profile and its latest healthy configuration. */
export const DESKTOP_PROFILE_ROLLBACK_PATH = '/api/desktop/profiles/rollback'

/** Renderer-safe projection of one discovered profile. */
export interface DesktopSettingsProfileView {
  /** Profile name accepted by the launcher. */
  readonly name: string
  /** Whether its manifest already exists on disk. */
  readonly exists: boolean
  /** Whether it contains the Web application required by Desktop. */
  readonly webCapable: boolean
  /** Whether the launcher can select it. */
  readonly selectable: boolean
  /** Whether the profile can be removed without affecting recovery state. */
  readonly deletable: boolean
}

/** Requested and generation-effective Market provider state. */
export interface DesktopSettingsMarketView {
  /** Explicit or fail-safe provider requested on disk. */
  readonly requested: DesktopMarketProvider
  /** Provider composed into the currently running generation. */
  readonly effective: DesktopMarketProvider
  /** Whether an absent or invalid legacy state produced the fail-safe default. */
  readonly legacyDefaulted: boolean
}

/** Complete renderer-safe Desktop settings state. */
export interface DesktopSettingsResponse {
  /** Profile backing the currently running generation. */
  readonly current: string
  /** Fresh profile discovery without filesystem paths or manifest details. */
  readonly profiles: readonly DesktopSettingsProfileView[]
  /** Market choice for the current and next generation. */
  readonly market: DesktopSettingsMarketView
}

/** Exact body accepted by the profile-creation endpoint. */
export interface DesktopProfileCreateRequest {
  readonly name: string
}

/** Successful creation returns a fresh state that includes the new profile. */
export type DesktopProfileCreateResponse = DesktopSettingsResponse

/** Exact body accepted by the profile-selection endpoint. */
export interface DesktopProfileSelectRequest {
  readonly name: string
}

/** Successful persisted selection returned before the Host restarts. */
export interface DesktopRestartAcceptance {
  readonly accepted: true
  readonly restartRequired: boolean
}

/** Successful profile selection handoff. */
export type DesktopProfileSelectResponse = DesktopRestartAcceptance

/** Exact body accepted by the profile-deletion endpoint. */
export interface DesktopProfileDeleteRequest {
  readonly name: string
}

/** Successful deletion returns a fresh state without the removed profile. */
export type DesktopProfileDeleteResponse = DesktopSettingsResponse

/** Exact body accepted by the Market-provider endpoint. */
export interface DesktopMarketSelectRequest {
  readonly provider: DesktopMarketProvider
}

/** Successful Market selection handoff. */
export type DesktopMarketSelectResponse = DesktopRestartAcceptance

/** Renderer-safe projection of one direct Profile bundle. */
export interface DesktopPluginBundleView {
  /** Generation-local opaque target identity. */
  readonly bundleId: string
  /** Informational package identity used only for display and classification. */
  readonly packageName: string
  /** Persisted state that will be used by the next generation. */
  readonly status: 'active' | 'disabled'
  /** Whether Desktop permits this exact bundle to be toggled. */
  readonly mutable: boolean
}

/** One actual non-group entry in the current Cordis Loader tree. */
export interface DesktopPluginEntryView {
  readonly entryId: string
  readonly moduleName: string
  /** Persisted state selected for the next generation after harness linking. */
  readonly enabled: boolean
  /** Harness or generic group owning this entry. */
  readonly harnessId: string
  /** Whether this Loader row is a harness AgentFactory. */
  readonly engine: boolean
  /** Whether the primary harness selection, not the switch, owns this state. */
  readonly locked: boolean
}

/** One built-in, common, or user-created harness group. */
export interface DesktopPluginHarnessView {
  readonly id: string
  readonly name: string
  readonly builtIn: boolean
  /** Loader row id of the AgentFactory owned by this harness; absent for common. */
  readonly engine?: string
  /** Whether this harness can own the primary AgentFactory selection. */
  readonly selectable: boolean
}

/** Current direct-bundle inventory plus whether it differs from the running generation. */
export interface DesktopPluginsResponse {
  readonly bundles: readonly DesktopPluginBundleView[]
  readonly entries: readonly DesktopPluginEntryView[]
  readonly harnesses: readonly DesktopPluginHarnessView[]
  /** Primary AgentFactory id fixed for the running generation. */
  readonly primaryHarness: string
  readonly restartRequired: boolean
}

/** Exact reversible toggle request accepted from the renderer. */
export interface DesktopPluginToggleRequest {
  readonly bundleId: string
  readonly enabled: boolean
}

export interface DesktopPluginEntryToggleRequest {
  readonly entryId: string
  readonly enabled: boolean
}

export interface DesktopHarnessCreateRequest {
  readonly name: string
  readonly engine?: string
}

export interface DesktopHarnessAssignRequest {
  readonly entryId: string
  readonly harnessId: string
}

export interface DesktopHarnessDeleteRequest {
  readonly harnessId: string
}

/** Fresh inventory returned after a persisted toggle. */
export interface DesktopPluginToggleResponse extends DesktopPluginsResponse {
  readonly accepted: true
}

export type DesktopPluginEntryToggleResponse = DesktopPluginToggleResponse
export type DesktopHarnessMutationResponse = DesktopPluginToggleResponse

/** Exact empty body accepted by the plugin restart endpoint. */
export type DesktopPluginRestartRequest = Readonly<Record<string, never>>

/** Restart handoff after pending plugin changes. */
export type DesktopPluginRestartResponse = DesktopRestartAcceptance

/** Exact empty body accepted by the terminal endpoint. */
export type DesktopTerminalOpenRequest = Readonly<Record<string, never>>

/** Successful handoff to the launcher-owned terminal action. */
export interface DesktopTerminalOpenResponse {
  readonly accepted: true
}

/** Exact empty body accepted by the diagnostic-export endpoint. */
export type DesktopDiagnosticsExportRequest = Readonly<Record<string, never>>

/** Successful handoff to the launcher-owned diagnostic export flow. */
export interface DesktopDiagnosticsExportResponse {
  readonly accepted: true
}

/** Exact empty body accepted by the native Profile-creator endpoint. */
export type DesktopProfileCreateWindowRequest = Readonly<Record<string, never>>

/** Successful handoff to the isolated native Profile creator. */
export interface DesktopProfileCreateWindowResponse {
  readonly accepted: true
}

/** Exact empty body accepted by the last-known-good rollback endpoint. */
export type DesktopProfileRollbackRequest = Readonly<Record<string, never>>

/** Persisted rollback handoff returned before the running Host is quiesced. */
export interface DesktopProfileRollbackResponse extends DesktopRestartAcceptance {
  readonly targetProfile: string
}

/** Stable API failure shape that never contains native paths or raw causes. */
export interface DesktopSettingsErrorResponse {
  readonly error: string
}
