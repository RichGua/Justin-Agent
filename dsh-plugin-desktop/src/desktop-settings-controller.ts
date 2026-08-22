/** Launcher-backed controller for the private Desktop settings API. */

import type {
  DesktopMarketProvider,
  DesktopMarketSnapshot,
} from './desktop-market.ts'
import type { DesktopProfileSummary } from './profile-manager.ts'
import type { DesktopProfiles } from './profile-service.ts'
import type { DesktopPlugins } from './desktop-plugins.ts'
import type { DesktopPluginEntriesService } from './desktop-plugin-entries.ts'
import type {
  DesktopMarketSelectResponse,
  DesktopDiagnosticsExportResponse,
  DesktopProfileCreateResponse,
  DesktopProfileCreateWindowResponse,
  DesktopProfileDeleteResponse,
  DesktopProfileRollbackResponse,
  DesktopProfileSelectResponse,
  DesktopPluginRestartResponse,
  DesktopPluginsResponse,
  DesktopPluginToggleResponse,
  DesktopSettingsMarketView,
  DesktopSettingsProfileView,
  DesktopSettingsResponse,
  DesktopTerminalOpenResponse,
} from './desktop-settings-contract.ts'

/** Launcher capabilities used without exposing their filesystem roots. */
export interface DesktopSettingsControllerBootstrap {
  /** Generation-scoped profile service. */
  readonly profiles: Pick<DesktopProfiles, 'current' | 'list' | 'create'>
    & Partial<Pick<DesktopProfiles, 'canDelete' | 'delete'>>
  /** Direct Profile bundle state and its restart-safe mutation capability. */
  readonly plugins: Pick<DesktopPlugins, 'list' | 'previewDisable' | 'executeDisable' | 'previewEnable' | 'executeEnable'>
  /** Effective Cordis Loader entries and Profile-local harness metadata. */
  readonly pluginEntries?: Pick<DesktopPluginEntriesService,
    'snapshot' | 'setEnabled' | 'createHarness' | 'assignHarness' | 'deleteHarness'>
  /** Persist one already-validated profile as pending without restarting. */
  persistProfileSelection(name: string): void | Promise<void>
  /** Read the latest persisted request and the startup-effective provider. */
  readMarket(): DesktopMarketSnapshot
  /** Persist an explicit provider request. */
  selectMarket(provider: DesktopMarketProvider): Promise<DesktopMarketSnapshot>
  /** Queue an orderly restart after a response confirms persisted selection. */
  scheduleRestart(): void
  /** Open the launcher-owned DSH terminal. */
  openTerminal(): void
  /** Export diagnostics through the launcher-owned privacy flow. */
  exportDiagnostics(): void | Promise<void>
  /** Open the isolated native Profile creator. */
  openProfileCreator(): void
  /** Prepare a last-known-good rollback without quiescing the Host yet. */
  prepareProfileRollback(): DesktopSettingsPostResponse<DesktopProfileRollbackResponse>
}

/** A persisted response plus work that must run only after `res.end()`. */
export interface DesktopSettingsPostResponse<T extends object> {
  readonly response: T
  readonly afterResponse?: () => void
}

/** Remove paths, bundle identities, and parser diagnostics from a profile. */
export function projectDesktopSettingsProfile(
  profile: DesktopProfileSummary,
  deletable = false,
): DesktopSettingsProfileView {
  return Object.freeze({
    name: profile.name,
    exists: profile.exists,
    webCapable: profile.webCapable,
    selectable: profile.webCapable && profile.problem === undefined,
    deletable,
  })
}

function projectMarket(
  value: DesktopMarketSnapshot,
  effective: DesktopMarketProvider,
): DesktopSettingsMarketView {
  return Object.freeze({
    requested: value.requested,
    effective,
    legacyDefaulted: value.legacyDefaulted,
  })
}

/**
 * Generation-scoped controller for Profile and Market preferences.
 *
 * The provider composed at startup remains `effective` for this controller's
 * lifetime. Persisting another provider changes only `requested` until the
 * queued restart creates a new Host generation.
 */
export class DesktopSettingsController {
  private readonly effectiveMarket: DesktopMarketProvider
  private readonly pluginBaseline: ReadonlyMap<string, 'active' | 'disabled'>

  constructor(private readonly bootstrap: DesktopSettingsControllerBootstrap) {
    this.effectiveMarket = bootstrap.readMarket().effective
    this.pluginBaseline = new Map(
      bootstrap.plugins.list().map(bundle => [bundle.packageName, bundle.status]),
    )
  }

  /** Read a fresh renderer-safe direct-bundle projection. */
  readPlugins(): DesktopPluginsResponse {
    const bundles = this.bootstrap.plugins.list().map(bundle => Object.freeze({
      bundleId: bundle.bundleId,
      packageName: bundle.packageName,
      status: bundle.status,
      mutable: bundle.mutable,
    }))
    const pluginEntries = this.bootstrap.pluginEntries?.snapshot()
    const entries = pluginEntries?.entries.map(entry => Object.freeze({
      entryId: entry.entryId,
      moduleName: entry.moduleName,
      enabled: entry.enabled,
      harnessId: entry.harnessId,
      engine: entry.engine,
      locked: entry.locked,
    })) ?? []
    const harnesses = pluginEntries?.harnesses.map(harness => Object.freeze({
      id: harness.id,
      name: harness.name,
      builtIn: harness.builtIn,
      ...(harness.engine === undefined ? {} : { engine: harness.engine }),
      selectable: harness.selectable,
    })) ?? []
    const restartRequired = bundles.some(bundle =>
      this.pluginBaseline.get(bundle.packageName) !== bundle.status)
      || (pluginEntries?.restartRequired ?? false)
    return Object.freeze({
      bundles: Object.freeze(bundles),
      entries: Object.freeze(entries),
      harnesses: Object.freeze(harnesses),
      primaryHarness: pluginEntries?.primaryHarness ?? 'deepseek',
      restartRequired,
    })
  }

  /** Persist one real Loader entry switch, excluding locked harness engines. */
  async setPluginEntryEnabled(entryId: string, enabled: boolean): Promise<DesktopPluginToggleResponse> {
    if (this.bootstrap.pluginEntries === undefined) {
      throw new Error('dsh-plugin-desktop: Loader plugin management is unavailable')
    }
    await this.bootstrap.pluginEntries.setEnabled(entryId, enabled)
    return Object.freeze({ accepted: true, ...this.readPlugins() })
  }

  /** Create one user harness, optionally owning one AgentFactory Loader row. */
  async createHarness(name: string, engine?: string): Promise<DesktopPluginToggleResponse> {
    if (this.bootstrap.pluginEntries === undefined) {
      throw new Error('dsh-plugin-desktop: Loader plugin management is unavailable')
    }
    await this.bootstrap.pluginEntries.createHarness(name, engine)
    return Object.freeze({ accepted: true, ...this.readPlugins() })
  }

  /** Move one entry into a harness or the common group without changing its Loader state. */
  async assignHarness(entryId: string, harnessId: string): Promise<DesktopPluginToggleResponse> {
    if (this.bootstrap.pluginEntries === undefined) {
      throw new Error('dsh-plugin-desktop: Loader plugin management is unavailable')
    }
    await this.bootstrap.pluginEntries.assignHarness(entryId, harnessId)
    return Object.freeze({ accepted: true, ...this.readPlugins() })
  }

  /** Delete only a user-created harness and release its entries to common. */
  async deleteHarness(harnessId: string): Promise<DesktopPluginToggleResponse> {
    if (this.bootstrap.pluginEntries === undefined) {
      throw new Error('dsh-plugin-desktop: Loader plugin management is unavailable')
    }
    await this.bootstrap.pluginEntries.deleteHarness(harnessId)
    return Object.freeze({ accepted: true, ...this.readPlugins() })
  }

  /** Persist one reversible bundle toggle after revalidating its opaque target. */
  async setPluginEnabled(bundleId: string, enabled: boolean): Promise<DesktopPluginToggleResponse> {
    const target = this.bootstrap.plugins.list().find(bundle => bundle.bundleId === bundleId)
    if (target === undefined) throw new Error('dsh-plugin-desktop: plugin target is unavailable')
    if (!target.mutable) throw new Error('dsh-plugin-desktop: plugin target is immutable')
    const wanted = enabled ? 'active' : 'disabled'
    if (target.status !== wanted) {
      if (enabled) {
        const preview = this.bootstrap.plugins.previewEnable(bundleId)
        await this.bootstrap.plugins.executeEnable(preview.previewId)
      } else {
        const preview = this.bootstrap.plugins.previewDisable(bundleId)
        await this.bootstrap.plugins.executeDisable(preview.previewId)
      }
    }
    return Object.freeze({ accepted: true, ...this.readPlugins() })
  }

  /** Restart only when persisted plugin state differs from this generation. */
  restartPlugins(): DesktopSettingsPostResponse<DesktopPluginRestartResponse> {
    const restartRequired = this.readPlugins().restartRequired
    return Object.freeze({
      response: Object.freeze({ accepted: true, restartRequired }),
      ...(restartRequired ? { afterResponse: () => { this.bootstrap.scheduleRestart() } } : {}),
    })
  }

  /** Read a fresh, renderer-safe settings projection. */
  read(): DesktopSettingsResponse {
    return Object.freeze({
      current: this.bootstrap.profiles.current.name,
      profiles: Object.freeze(
        this.bootstrap.profiles.list().map(profile => projectDesktopSettingsProfile(
          profile,
          this.bootstrap.profiles.canDelete?.(profile.name) ?? false,
        )),
      ),
      market: projectMarket(this.bootstrap.readMarket(), this.effectiveMarket),
    })
  }

  /** Create one safe profile without selecting it or requesting restart. */
  createProfile(name: string): DesktopProfileCreateResponse {
    this.bootstrap.profiles.create(name)
    return this.read()
  }

  /** Delete one inactive user profile and return the fresh settings state. */
  async deleteProfile(name: string): Promise<DesktopProfileDeleteResponse> {
    if (this.bootstrap.profiles.delete === undefined) {
      throw new Error('dsh-plugin-desktop: profile deletion is unavailable')
    }
    await this.bootstrap.profiles.delete(name)
    return this.read()
  }

  /** Persist a fresh compatible profile, deferring restart until after response. */
  async selectProfile(
    name: string,
  ): Promise<DesktopSettingsPostResponse<DesktopProfileSelectResponse>> {
    const restartRequired = name !== this.bootstrap.profiles.current.name
    if (restartRequired) {
      const profile = this.bootstrap.profiles.list().find(candidate => candidate.name === name)
      if (profile === undefined || !profile.webCapable || profile.problem !== undefined) {
        throw new Error(`dsh-plugin-desktop: profile ${JSON.stringify(name)} is not selectable`)
      }
      await this.bootstrap.persistProfileSelection(name)
    }
    return Object.freeze({
      response: Object.freeze({ accepted: true, restartRequired }),
      ...(restartRequired ? { afterResponse: () => { this.bootstrap.scheduleRestart() } } : {}),
    })
  }

  /** Persist a provider and defer restart until after the response is ended. */
  async selectMarket(
    provider: DesktopMarketProvider,
  ): Promise<DesktopSettingsPostResponse<DesktopMarketSelectResponse>> {
    await this.bootstrap.selectMarket(provider)
    const restartRequired = provider !== this.effectiveMarket
    return Object.freeze({
      response: Object.freeze({ accepted: true, restartRequired }),
      ...(restartRequired ? { afterResponse: () => { this.bootstrap.scheduleRestart() } } : {}),
    })
  }

  /** Open the native terminal through the launcher-owned action. */
  openTerminal(): DesktopTerminalOpenResponse {
    this.bootstrap.openTerminal()
    return Object.freeze({ accepted: true })
  }

  /** Export diagnostics through the native confirmation and reveal flow. */
  async exportDiagnostics(): Promise<DesktopDiagnosticsExportResponse> {
    await this.bootstrap.exportDiagnostics()
    return Object.freeze({ accepted: true })
  }

  /** Open the native creator that creates, selects, and restarts safely. */
  openProfileCreator(): DesktopProfileCreateWindowResponse {
    this.bootstrap.openProfileCreator()
    return Object.freeze({ accepted: true })
  }

  /** Hand off a validated rollback that starts only after the HTTP response. */
  rollbackProfile(): DesktopSettingsPostResponse<DesktopProfileRollbackResponse> {
    return this.bootstrap.prepareProfileRollback()
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** Launcher-owned controller behind the private loopback settings API. */
    desktopSettingsController: DesktopSettingsController
  }
}

export default DesktopSettingsController
