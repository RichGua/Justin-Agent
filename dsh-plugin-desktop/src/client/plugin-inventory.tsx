import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react'
import type { ClientContext } from '@deepseek-ai/dsh-client-runtime/client'
import type { SettingsScope } from '@deepseek-ai/dsh-client-runtime/client'
import type {} from '@deepseek-ai/dsh-client-locale/client'
import type {} from '@deepseek-ai/dsh-client-ui-settings-plugins/client'
import type { PropsLocale, PropsRuntime, InjectFace } from '@deepseek-ai/dsh-client-ui-slots'
import {
  createDesktopSettingsApi,
  type DesktopPluginEntryView,
  type DesktopPluginHarnessView,
  type DesktopPluginsView,
  type DesktopSettingsApi,
} from './desktop-settings-api.ts'
import type { DesktopShellSettings } from './DesktopSettingsSection.tsx'

const NS = 'rundeep.pluginInventory'

const zh = {
  tab: '按 Harness 分类', loading: '正在读取 Loader 插件…', error: '暂时无法读取插件列表。', retry: '重试', search: '搜索插件',
  noPlugins: '此分组中没有插件。', enabled: '已启用', disabled: '已停用', enable: '启用', disable: '停用',
  changing: '正在保存插件状态…', changeError: '插件状态保存失败，请重试。',
  restartRequired: '插件状态已保存，需要重启 RunDeep 才会生效。', restart: '立即重启', restarting: '正在重启…', restartError: '无法重启 RunDeep，请手动重启应用。',
  expand: '展开', collapse: '折叠',
  engineSwitch: 'Harness 运行引擎', engineSwitchHint: '切换后自动重启，并只加载该引擎对应的插件集；通用插件不受影响。',
  newHarness: '新增 Harness', harnessName: 'Harness 名称', enginePick: '引擎插件（可选）', createHarness: '添加', deleteHarness: '删除 Harness', harnessError: 'Harness 保存失败，请重试。',
  move: '移动归属', engine: '引擎', primary: '当前主引擎', common: '通用插件', commonHint: '通用插件不随 Harness 切换改变', lockedHint: '由引擎设置控制', lockedOff: '随主引擎停用',
} as const
type LocaleKey = keyof typeof zh
const en: Record<LocaleKey, string> = {
  tab: 'By Harness', loading: 'Reading Loader plugins…', error: 'The plugin list is temporarily unavailable.', retry: 'Retry', search: 'Search plugins',
  noPlugins: 'This group has no plugins.', enabled: 'Enabled', disabled: 'Disabled', enable: 'Enable', disable: 'Disable',
  changing: 'Saving plugin state…', changeError: 'The plugin state could not be saved. Try again.',
  restartRequired: 'The plugin state was saved. Restart RunDeep to apply it.', restart: 'Restart now', restarting: 'Restarting…', restartError: 'RunDeep could not restart. Restart the app manually.',
  expand: 'Expand', collapse: 'Collapse',
  engineSwitch: 'Harness runtime', engineSwitchHint: 'Switching restarts automatically and loads only that engine’s plugin set; common plugins are untouched.',
  newHarness: 'New harness', harnessName: 'Harness name', enginePick: 'Engine plugin (optional)', createHarness: 'Add', deleteHarness: 'Delete harness', harnessError: 'The harness could not be saved. Try again.',
  move: 'Move to group', engine: 'Engine', primary: 'Active primary harness', common: 'Common plugins', commonHint: 'Not affected by harness switches', lockedHint: 'Controlled by the harness setting', lockedOff: 'Off with the primary harness',
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap { 'rundeep.pluginInventory': LocaleKey }
}

interface Injected {
  api: Pick<DesktopSettingsApi,
    | 'readPlugins'
    | 'setPluginEntryEnabled'
    | 'createHarness'
    | 'assignHarness'
    | 'deleteHarness'
    | 'restartPlugins'>
  desktopSettings: SettingsScope<DesktopShellSettings>
}
type Props = PropsRuntime<'settings.plugins.tab'> & PropsLocale<typeof NS> & InjectFace<Injected>

function shortName(moduleName: string) {
  return moduleName
    .replace(/^@deepseek-ai\/dsh-/u, '')
    .replace(/^@justin-agent\/dsh-harness-/u, '')
    .replace(/^dsh-plugin-desktop\//u, '')
}

/** Built-in group names are localized; custom harnesses keep their own names. */
function groupName(harness: DesktopPluginHarnessView, common: string): string {
  if (harness.id === 'common') return common
  return harness.name
}

/** Cordis Loader inventory grouped by harness: real switches with linked sets. */
export function HarnessPluginInventory({ api, t, desktopSettings }: Props) {
  const [plugins, setPlugins] = useState<DesktopPluginsView>()
  const [failed, setFailed] = useState(false)
  const [query, setQuery] = useState('')
  const [pendingEntry, setPendingEntry] = useState<string>()
  const [pendingHarness, setPendingHarness] = useState(false)
  const [changeFailed, setChangeFailed] = useState(false)
  const [harnessFailed, setHarnessFailed] = useState(false)
  const [newHarness, setNewHarness] = useState('')
  const [newEngine, setNewEngine] = useState('')
  const [restarting, setRestarting] = useState(false)
  const [restartFailed, setRestartFailed] = useState(false)
  const [engineBusy, setEngineBusy] = useState(false)
  const [openMenu, setOpenMenu] = useState<string>()
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const load = useCallback(async () => {
    setFailed(false)
    try { setPlugins(await api.readPlugins()) } catch { setFailed(true) }
  }, [api])
  useEffect(() => { void load() }, [load])

  const groups = useMemo(() => {
    const result = new Map<string, DesktopPluginEntryView[]>()
    for (const harness of plugins?.harnesses ?? []) result.set(harness.id, [])
    const needle = query.trim().toLocaleLowerCase()
    for (const entry of plugins?.entries ?? []) {
      if (needle && !entry.moduleName.toLocaleLowerCase().includes(needle)) continue
      result.get(entry.harnessId)?.push(entry)
    }
    return result
  }, [plugins, query])

  const engineCandidates = useMemo(() => {
    if (plugins === undefined) return []
    const owned = new Set(plugins.harnesses.flatMap(harness =>
      harness.engine === undefined ? [] : [harness.engine]))
    return plugins.entries
      .filter(entry => !entry.engine && !owned.has(entry.entryId))
      .map(entry => ({ entryId: entry.entryId, moduleName: entry.moduleName }))
  }, [plugins])

  const switchEngine = async (next: string): Promise<void> => {
    if (engineBusy || plugins === undefined || next === plugins.primaryHarness) return
    setChangeFailed(false)
    setEngineBusy(true)
    setRestarting(true)
    try {
      await desktopSettings.set('harness', next)
    } catch {
      setChangeFailed(true)
      setRestarting(false)
    } finally {
      setEngineBusy(false)
    }
  }

  const toggle = async (entry: DesktopPluginEntryView) => {
    if (pendingEntry !== undefined || pendingHarness || entry.locked) return
    setChangeFailed(false)
    setRestartFailed(false)
    setPendingEntry(entry.entryId)
    try { setPlugins(await api.setPluginEntryEnabled(entry.entryId, !entry.enabled)) } catch { setChangeFailed(true) } finally { setPendingEntry(undefined) }
  }

  const assign = async (entryId: string, harnessId: string) => {
    if (pendingEntry !== undefined || pendingHarness) return
    setHarnessFailed(false)
    setOpenMenu(undefined)
    setPendingEntry(entryId)
    try {
      setPlugins(await api.assignHarness(entryId, harnessId))
      setExpanded(current => ({ ...current, [harnessId]: true }))
    } catch { setHarnessFailed(true) } finally { setPendingEntry(undefined) }
  }

  const createHarness = async (event: FormEvent) => {
    event.preventDefault()
    const name = newHarness.trim()
    if (name.length === 0 || pendingHarness) return
    setHarnessFailed(false)
    setPendingHarness(true)
    try {
      const next = await api.createHarness(name, newEngine === '' ? undefined : newEngine)
      setPlugins(next)
      const created = next.harnesses.find(harness => harness.name === name)
      if (created !== undefined) setExpanded(current => ({ ...current, [created.id]: true }))
      setNewHarness('')
      setNewEngine('')
    } catch { setHarnessFailed(true) } finally { setPendingHarness(false) }
  }

  const deleteHarness = async (harnessId: string) => {
    if (pendingHarness || pendingEntry !== undefined) return
    setHarnessFailed(false)
    setPendingHarness(true)
    try { setPlugins(await api.deleteHarness(harnessId)) } catch { setHarnessFailed(true) } finally { setPendingHarness(false) }
  }

  const restart = async () => {
    if (restarting) return
    setRestartFailed(false)
    setRestarting(true)
    try { await api.restartPlugins() } catch { setRestartFailed(true); setRestarting(false) }
  }

  if (plugins === undefined && !failed) return <p className="rundeepInventoryStatus">{t('loading')}</p>
  if (failed) return <div className="rundeepInventoryStatus"><p role="alert">{t('error')}</p><button type="button" onClick={() => { void load() }}>{t('retry')}</button></div>
  if (plugins === undefined) return null

  const selectable = plugins.harnesses.filter(harness => harness.selectable)
  const primary = plugins.primaryHarness

  return <section className="rundeepInventory">
    <div className="rundeepInventoryTools">
      <label className="rundeepInventorySearch"><span>{t('search')}</span><input type="search" value={query} onChange={event => setQuery(event.currentTarget.value)} placeholder={t('search')} /></label>
      <form className="rundeepInventoryHarnessForm" onSubmit={createHarness}>
        <label><span>{t('newHarness')}</span><input value={newHarness} maxLength={64} placeholder={t('harnessName')} onChange={event => setNewHarness(event.currentTarget.value)} /></label>
        <label><span>{t('enginePick')}</span><select value={newEngine} onChange={event => setNewEngine(event.currentTarget.value)}>
          <option value="">—</option>
          {engineCandidates.map(candidate => <option key={candidate.entryId} value={candidate.entryId}>{shortName(candidate.moduleName)}</option>)}
        </select></label>
        <button type="submit" disabled={newHarness.trim().length === 0 || pendingHarness}>{t('createHarness')}</button>
      </form>
    </div>

    <section className="rundeepInventoryEngine" aria-label={t('engineSwitch')}>
      <span className="rundeepInventoryEngineLabel">{t('engineSwitch')}</span>
      <div className="rundeepInventorySegmented" role="radiogroup" aria-label={t('engineSwitch')}>
        {selectable.map(harness => {
          const current = harness.id === primary
          return <button
            key={harness.id}
            type="button"
            role="radio"
            aria-checked={current}
            disabled={engineBusy || restarting}
            className={current ? 'rundeepInventorySegmentActive' : 'rundeepInventorySegment'}
            onClick={() => { void switchEngine(harness.id) }}
          >
            {groupName(harness, t('common'))}
          </button>
        })}
      </div>
      <p className="rundeepInventoryHint">{t('engineSwitchHint')}</p>
    </section>

    {restarting ? <div className="rundeepInventoryRestart" role="status"><span>{t('restarting')}</span></div> : null}
    {restartFailed ? <p className="rundeepInventoryError" role="alert">{t('restartError')}</p> : null}
    {changeFailed ? <p className="rundeepInventoryError" role="alert">{t('changeError')}</p> : null}
    {harnessFailed ? <p className="rundeepInventoryError" role="alert">{t('harnessError')}</p> : null}
    {pendingEntry !== undefined ? <p className="rundeepInventorySaving" role="status">{t('changing')}</p> : null}

    {plugins.restartRequired ? <div className="rundeepInventoryRestart" role="status"><span>{t('restartRequired')}</span><button type="button" disabled={restarting} onClick={() => { void restart() }}>{t('restart')}</button></div> : null}
    <p className="rundeepInventoryHint">{t('commonHint')}</p>
    {plugins.harnesses.map(harness => {
      const open = expanded[harness.id] ?? true
      const entries = groups.get(harness.id) ?? []
      const current = harness.id === primary
      return <section className="rundeepInventoryGroup" key={harness.id}>
        <div className="rundeepInventoryGroupHeading">
          <button type="button" className="rundeepInventoryGroupHeader" aria-expanded={open} aria-label={`${t(open ? 'collapse' : 'expand')}: ${groupName(harness, t('common'))}`} onClick={() => setExpanded(currentState => ({ ...currentState, [harness.id]: !open }))}>
            <span>{groupName(harness, t('common'))}</span>
            {current ? <b className="rundeepInventoryPrimary">{t('primary')}</b> : null}
            {harness.engine !== undefined ? <i className="rundeepInventoryEngine">{t('engine')}</i> : null}
            <small>{entries.length}</small><i aria-hidden="true">⌄</i>
          </button>
          {!harness.builtIn ? <button type="button" className="rundeepInventoryDeleteHarness" aria-label={`${t('deleteHarness')}: ${harness.name}`} disabled={pendingHarness || pendingEntry !== undefined || current} onClick={() => { void deleteHarness(harness.id) }}>{t('deleteHarness')}</button> : null}
        </div>
        {open ? <div className="rundeepInventoryGroupBody">
          {entries.length === 0 ? <p>{t('noPlugins')}</p> : <ul>
            {entries.map(entry => {
              const menuOpen = openMenu === entry.entryId
              const targets = plugins.harnesses.filter(option => option.id !== entry.harnessId)
              const stateLabel = entry.engine
                ? t('lockedHint')
                : entry.locked
                  ? t('lockedOff')
                  : t(entry.enabled ? 'enabled' : 'disabled')
              return <li key={entry.entryId}>
                <span className="rundeepInventoryName" title={`${entry.moduleName} · ${entry.entryId}`}>{shortName(entry.moduleName)}{entry.engine ? <b className="rundeepInventoryEngineBadge">{t('engine')}</b> : null}</span>
                <span className={`rundeepInventoryState${entry.enabled ? '' : ' rundeepInventoryStateOff'}`}>{stateLabel}</span>
                {!entry.engine ? (
                  <div className="rundeepInventoryMenu">
                    <button type="button" className="rundeepInventoryMenuButton" aria-label={`${t('move')}: ${shortName(entry.moduleName)}`} aria-expanded={menuOpen} disabled={pendingEntry !== undefined || pendingHarness} onClick={() => setOpenMenu(menuOpen ? undefined : entry.entryId)}>⋯</button>
                    {menuOpen ? <div className="rundeepInventoryMenuList" role="menu">
                      {targets.map(option => <button type="button" role="menuitem" key={option.id} onClick={() => { void assign(entry.entryId, option.id) }}>{groupName(option, t('common'))}</button>)}
                    </div> : null}
                  </div>
                ) : <span className="rundeepInventoryMenuSpacer" />}
                <button type="button" className="rundeepInventorySwitch" role="switch" aria-checked={entry.enabled} aria-label={`${t(entry.enabled ? 'disable' : 'enable')}: ${shortName(entry.moduleName)}`} disabled={pendingEntry !== undefined || pendingHarness || entry.locked} onClick={() => { void toggle(entry) }}><span /></button>
              </li>
            })}
          </ul>}
        </div> : null}
      </section>
    })}
  </section>
}

function installStyles() {
  const style = document.createElement('style')
  style.textContent = `
.rundeepInventory{display:flex;flex-direction:column;gap:14px}
.rundeepInventoryTools{display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,1.2fr);gap:12px;align-items:end}
.rundeepInventorySearch,.rundeepInventoryHarnessForm label{display:flex;gap:10px;align-items:center}.rundeepInventorySearch span,.rundeepInventoryHarnessForm label span{font-size:13px;color:var(--dsw-alias-label-tertiary)}.rundeepInventorySearch input,.rundeepInventoryHarnessForm input,.rundeepInventoryHarnessForm select{min-width:0;flex:1;border:1px solid var(--dsw-alias-border-l2);border-radius:8px;background:var(--dsw-alias-bg-layer-3);color:inherit;padding:8px 10px}.rundeepInventoryHarnessForm{display:grid;grid-template-columns:minmax(0,1fr) minmax(160px,.7fr) auto;gap:8px;align-items:end}.rundeepInventoryHarnessForm button,.rundeepInventoryDeleteHarness{border:1px solid var(--dsw-alias-border-l2);border-radius:7px;background:var(--dsw-alias-bg-layer-3);color:inherit;padding:7px 10px;cursor:pointer}
.rundeepInventoryEngine{display:flex;flex-direction:column;gap:8px;padding:14px 16px;border:1px solid var(--dsw-alias-border-l2);border-radius:12px;background:var(--dsw-alias-bg-layer-2)}.rundeepInventoryEngineLabel{font-size:13px;font-weight:650}.rundeepInventorySegmented{display:flex;gap:6px;padding:4px;border-radius:10px;background:var(--dsw-alias-bg-layer-3)}.rundeepInventorySegment,.rundeepInventorySegmentActive{flex:1;border:0;border-radius:8px;padding:9px 12px;font:inherit;font-size:13px;font-weight:600;color:var(--dsw-alias-label-tertiary);background:transparent;cursor:pointer}.rundeepInventorySegmentActive{background:var(--dsw-alias-bg-layer-1);color:inherit;box-shadow:0 1px 3px rgba(0,0,0,.18)}.rundeepInventorySegment:disabled,.rundeepInventorySegmentActive:disabled{opacity:.5;cursor:not-allowed}
.rundeepInventoryHint{margin:0;font-size:12px;color:var(--dsw-alias-label-tertiary)}
.rundeepInventoryGroup{overflow:hidden;border:1px solid var(--dsw-alias-border-l2);border-radius:12px;background:var(--dsw-alias-bg-layer-2)}.rundeepInventoryGroupHeading{display:flex;align-items:center}.rundeepInventoryGroupHeader{display:flex;min-width:0;flex:1;align-items:center;gap:8px;border:0;background:transparent;color:inherit;padding:13px 15px;text-align:left;font:inherit;font-size:13px;font-weight:650;cursor:pointer}.rundeepInventoryGroupHeader small{margin-left:auto;color:var(--dsw-alias-label-tertiary);font-weight:500}.rundeepInventoryGroupHeader i{font-style:normal;transition:transform .16s ease}.rundeepInventoryGroupHeader[aria-expanded="true"] i{transform:rotate(180deg)}
.rundeepInventoryPrimary{border-radius:999px;background:rgba(79,124,255,.16);color:#4f7cff;font-size:11px;font-weight:650;padding:2px 8px}.rundeepInventoryEngine{color:var(--dsw-alias-label-tertiary);font-size:11px;font-weight:500;border:1px solid var(--dsw-alias-border-l2);border-radius:999px;padding:2px 8px}.rundeepInventoryEngineBadge{color:var(--dsw-alias-label-tertiary);font-size:10px;font-weight:650;border:1px solid var(--dsw-alias-border-l2);border-radius:4px;margin-left:6px;padding:1px 5px}.rundeepInventoryDeleteHarness{margin-right:10px;padding:5px 8px;font-size:12px;color:var(--dsw-alias-label-tertiary)}
.rundeepInventoryGroupBody{border-top:1px solid var(--dsw-alias-border-l2);padding:8px}.rundeepInventoryGroupBody>p{margin:0;padding:12px;color:var(--dsw-alias-label-tertiary);font-size:13px}.rundeepInventoryGroup ul{display:flex;flex-direction:column;gap:4px;margin:0;padding:0;list-style:none}.rundeepInventoryGroup li{position:relative;display:grid;grid-template-columns:minmax(140px,1fr) auto 32px 38px;align-items:center;gap:10px;padding:9px 10px;border-radius:9px;background:var(--dsw-alias-bg-layer-3);font-size:13px}.rundeepInventoryName{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}.rundeepInventoryState{color:var(--dsw-alias-label-tertiary);font-size:12px;white-space:nowrap}.rundeepInventoryStateOff{color:var(--dsw-alias-label-error,#e45c5c)}
.rundeepInventoryMenu{position:relative}.rundeepInventoryMenuButton{width:30px;height:26px;border:1px solid transparent;border-radius:7px;background:transparent;color:var(--dsw-alias-label-tertiary);font-size:15px;line-height:1;cursor:pointer}.rundeepInventoryMenuButton:hover,.rundeepInventoryMenuButton[aria-expanded="true"]{border-color:var(--dsw-alias-border-l2);background:var(--dsw-alias-bg-layer-2)}.rundeepInventoryMenuList{position:absolute;z-index:20;right:0;top:calc(100% + 4px);min-width:160px;display:flex;flex-direction:column;gap:2px;padding:5px;border:1px solid var(--dsw-alias-border-l2);border-radius:9px;background:var(--dsw-alias-bg-layer-1);box-shadow:0 6px 20px rgba(0,0,0,.22)}.rundeepInventoryMenuList button{border:0;border-radius:6px;background:transparent;color:inherit;font:inherit;font-size:12px;text-align:left;padding:7px 9px;cursor:pointer}.rundeepInventoryMenuList button:hover{background:rgba(79,124,255,.14)}.rundeepInventoryMenuSpacer{width:30px}
.rundeepInventorySwitch{position:relative;width:36px;height:20px;border:0;border-radius:999px;background:var(--dsw-alias-fill-tertiary);padding:2px;cursor:pointer;transition:background .16s ease}.rundeepInventorySwitch span{display:block;width:16px;height:16px;border-radius:50%;background:white;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .16s ease}.rundeepInventorySwitch[aria-checked="true"]{background:#4f7cff}.rundeepInventorySwitch[aria-checked="true"] span{transform:translateX(16px)}.rundeepInventorySwitch:disabled,.rundeepInventoryHarnessForm button:disabled,.rundeepInventoryDeleteHarness:disabled,.rundeepInventoryMenuButton:disabled{cursor:not-allowed;opacity:.38}
.rundeepInventoryRestart{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid rgba(79,124,255,.42);border-radius:9px;background:rgba(79,124,255,.10);padding:10px 12px;font-size:13px}.rundeepInventoryRestart button,.rundeepInventoryStatus button{border:0;border-radius:7px;background:#4f7cff;color:white;padding:6px 10px;cursor:pointer}.rundeepInventoryRestart button:disabled{opacity:.6}.rundeepInventoryError{margin:0;color:var(--dsw-alias-label-error,#e45c5c);font-size:13px}.rundeepInventorySaving,.rundeepInventoryStatus{margin:0;color:var(--dsw-alias-label-tertiary);font-size:13px}@media(max-width:760px){.rundeepInventoryTools,.rundeepInventoryHarnessForm{grid-template-columns:1fr}.rundeepInventoryGroup li{grid-template-columns:minmax(0,1fr) 38px}.rundeepInventoryState,.rundeepInventoryMenu{display:none}}
`
  document.head.append(style)
  return () => style.remove()
}

/** Replace the upstream tab body with the launcher-managed effective Loader view. */
export function applyRunDeepPluginInventory(ctx: ClientContext) {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'rundeep: plugin inventory dictionaries')
  ctx.effect(installStyles, 'rundeep: plugin inventory styles')
  const api = createDesktopSettingsApi()
  const desktopSettings = ctx.settingsScope.bind<DesktopShellSettings>({
    namespace: 'dsh-desktop',
  })
  ctx.slots.inject('settings.plugins.tab', () => ctx.slots.register({
    name: 'settings.plugins.tab', id: 'all', order: 10, label: () => ctx.locale.bind(NS)('tab'), locale: NS, inject: () => ({ api, desktopSettings }),
  }, HarnessPluginInventory))
}
