import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react'
import type { ClientContext } from '@deepseek-ai/dsh-client-runtime/client'
import type {} from '@deepseek-ai/dsh-client-locale/client'
import type {} from '@deepseek-ai/dsh-client-ui-settings-plugins/client'
import type { PropsLocale, PropsRuntime, InjectFace } from '@deepseek-ai/dsh-client-ui-slots'
import {
  createDesktopSettingsApi,
  type DesktopPluginEntryView,
  type DesktopPluginsView,
  type DesktopSettingsApi,
} from './desktop-settings-api.ts'

const NS = 'rundeep.pluginInventory'

const zh = {
  tab: '按 Harness 分类', loading: '正在读取 Loader 插件…', error: '暂时无法读取插件列表。', retry: '重试', search: '搜索插件',
  noPlugins: '此分类中没有插件。', enabled: '已启用', disabled: '已停用', enable: '启用', disable: '停用',
  changing: '正在保存插件状态…', changeError: '插件状态保存失败，请重试。',
  restartRequired: '插件状态已保存，需要重启 RunDeep 才会生效。', restart: '立即重启', restarting: '正在重启…', restartError: '无法重启 RunDeep，请手动重启应用。',
  expand: '展开', collapse: '折叠', category: '分类', newCategory: '新增分类', categoryName: '分类名称', createCategory: '添加', deleteCategory: '删除分类', categoryError: '分类保存失败，请重试。',
} as const
type LocaleKey = keyof typeof zh
const en: Record<LocaleKey, string> = {
  tab: 'By Harness', loading: 'Reading Loader plugins…', error: 'The plugin list is temporarily unavailable.', retry: 'Retry', search: 'Search plugins',
  noPlugins: 'This category has no plugins.', enabled: 'Enabled', disabled: 'Disabled', enable: 'Enable', disable: 'Disable',
  changing: 'Saving plugin state…', changeError: 'The plugin state could not be saved. Try again.',
  restartRequired: 'The plugin state was saved. Restart RunDeep to apply it.', restart: 'Restart now', restarting: 'Restarting…', restartError: 'RunDeep could not restart. Restart the app manually.',
  expand: 'Expand', collapse: 'Collapse', category: 'Category', newCategory: 'New category', categoryName: 'Category name', createCategory: 'Add', deleteCategory: 'Delete category', categoryError: 'The category could not be saved. Try again.',
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap { 'rundeep.pluginInventory': LocaleKey }
}

interface Injected {
  api: Pick<DesktopSettingsApi,
    | 'readPlugins'
    | 'setPluginEntryEnabled'
    | 'createPluginCategory'
    | 'assignPluginCategory'
    | 'deletePluginCategory'
    | 'restartPlugins'>
}
type Props = PropsRuntime<'settings.plugins.tab'> & PropsLocale<typeof NS> & InjectFace<Injected>

/** Only the primary Codex SDK adapter belongs to the default Codex category. */
export function pluginFamilyOf(moduleName: string): 'deepseek' | 'codex' {
  return moduleName === 'dsh-plugin-desktop/codex-harness' ? 'codex' : 'deepseek'
}

function shortName(moduleName: string) {
  return moduleName
    .replace(/^@deepseek-ai\/dsh-/u, '')
    .replace(/^@justin-agent\/dsh-harness-/u, '')
    .replace(/^dsh-plugin-desktop\//u, '')
}

/** Cordis Loader inventory: every row is a real switch; categories are metadata only. */
export function HarnessPluginInventory({ api, t }: Props) {
  const [plugins, setPlugins] = useState<DesktopPluginsView>()
  const [failed, setFailed] = useState(false)
  const [query, setQuery] = useState('')
  const [pendingEntry, setPendingEntry] = useState<string>()
  const [pendingCategory, setPendingCategory] = useState(false)
  const [changeFailed, setChangeFailed] = useState(false)
  const [categoryFailed, setCategoryFailed] = useState(false)
  const [newCategory, setNewCategory] = useState('')
  const [restarting, setRestarting] = useState(false)
  const [restartFailed, setRestartFailed] = useState(false)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ deepseek: true, codex: true })

  const load = useCallback(async () => {
    setFailed(false)
    try { setPlugins(await api.readPlugins()) } catch { setFailed(true) }
  }, [api])
  useEffect(() => { void load() }, [load])

  const groups = useMemo(() => {
    const result = new Map<string, DesktopPluginEntryView[]>()
    for (const category of plugins?.categories ?? []) result.set(category.id, [])
    const needle = query.trim().toLocaleLowerCase()
    for (const entry of plugins?.entries ?? []) {
      if (needle && !entry.moduleName.toLocaleLowerCase().includes(needle)) continue
      result.get(entry.categoryId)?.push(entry)
    }
    return result
  }, [plugins, query])

  const toggle = async (entry: DesktopPluginEntryView) => {
    if (pendingEntry !== undefined || pendingCategory) return
    setChangeFailed(false)
    setRestartFailed(false)
    setPendingEntry(entry.entryId)
    try { setPlugins(await api.setPluginEntryEnabled(entry.entryId, !entry.enabled)) } catch { setChangeFailed(true) } finally { setPendingEntry(undefined) }
  }

  const assign = async (entryId: string, categoryId: string) => {
    if (pendingEntry !== undefined || pendingCategory) return
    setCategoryFailed(false)
    setPendingEntry(entryId)
    try {
      setPlugins(await api.assignPluginCategory(entryId, categoryId))
      setExpanded(current => ({ ...current, [categoryId]: true }))
    } catch { setCategoryFailed(true) } finally { setPendingEntry(undefined) }
  }

  const createCategory = async (event: FormEvent) => {
    event.preventDefault()
    const name = newCategory.trim()
    if (name.length === 0 || pendingCategory) return
    setCategoryFailed(false)
    setPendingCategory(true)
    try {
      const next = await api.createPluginCategory(name)
      setPlugins(next)
      const created = next.categories.find(category => category.name === name)
      if (created !== undefined) setExpanded(current => ({ ...current, [created.id]: true }))
      setNewCategory('')
    } catch { setCategoryFailed(true) } finally { setPendingCategory(false) }
  }

  const deleteCategory = async (categoryId: string) => {
    if (pendingCategory || pendingEntry !== undefined) return
    setCategoryFailed(false)
    setPendingCategory(true)
    try { setPlugins(await api.deletePluginCategory(categoryId)) } catch { setCategoryFailed(true) } finally { setPendingCategory(false) }
  }

  const restart = async () => {
    if (restarting) return
    setRestartFailed(false)
    setRestarting(true)
    try { await api.restartPlugins() } catch { setRestartFailed(true); setRestarting(false) }
  }

  if (plugins === undefined && !failed) return <p className="rundeepInventoryStatus">{t('loading')}</p>
  if (failed) return <div className="rundeepInventoryStatus"><p role="alert">{t('error')}</p><button type="button" onClick={() => { void load() }}>{t('retry')}</button></div>

  return <section className="rundeepInventory">
    {plugins?.restartRequired ? <div className="rundeepInventoryRestart" role="status"><span>{t('restartRequired')}</span><button type="button" disabled={restarting} onClick={() => { void restart() }}>{t(restarting ? 'restarting' : 'restart')}</button></div> : null}
    {restartFailed ? <p className="rundeepInventoryError" role="alert">{t('restartError')}</p> : null}
    {changeFailed ? <p className="rundeepInventoryError" role="alert">{t('changeError')}</p> : null}
    {categoryFailed ? <p className="rundeepInventoryError" role="alert">{t('categoryError')}</p> : null}
    {pendingEntry !== undefined ? <p className="rundeepInventorySaving" role="status">{t('changing')}</p> : null}
    <div className="rundeepInventoryTools">
      <label className="rundeepInventorySearch"><span>{t('search')}</span><input type="search" value={query} onChange={event => setQuery(event.currentTarget.value)} placeholder={t('search')} /></label>
      <form className="rundeepInventoryCategoryForm" onSubmit={createCategory}>
        <label><span>{t('newCategory')}</span><input value={newCategory} maxLength={64} placeholder={t('categoryName')} onChange={event => setNewCategory(event.currentTarget.value)} /></label>
        <button type="submit" disabled={newCategory.trim().length === 0 || pendingCategory}>{t('createCategory')}</button>
      </form>
    </div>
    {plugins?.categories.map(category => {
      const open = expanded[category.id] ?? true
      const entries = groups.get(category.id) ?? []
      return <section className="rundeepInventoryGroup" key={category.id}>
        <div className="rundeepInventoryGroupHeading">
          <button type="button" className="rundeepInventoryGroupHeader" aria-expanded={open} aria-label={`${t(open ? 'collapse' : 'expand')}: ${category.name}`} onClick={() => setExpanded(current => ({ ...current, [category.id]: !open }))}>
            <span>{category.name}</span><small>{entries.length}</small><i aria-hidden="true">⌄</i>
          </button>
          {!category.builtIn ? <button type="button" className="rundeepInventoryDeleteCategory" aria-label={`${t('deleteCategory')}: ${category.name}`} disabled={pendingCategory || pendingEntry !== undefined} onClick={() => { void deleteCategory(category.id) }}>{t('deleteCategory')}</button> : null}
        </div>
        {open ? <div className="rundeepInventoryGroupBody">
          {entries.length === 0 ? <p>{t('noPlugins')}</p> : <ul>
            {entries.map(entry => <li key={entry.entryId}>
              <span className="rundeepInventoryName" title={`${entry.moduleName} · ${entry.entryId}`}>{shortName(entry.moduleName)}</span>
              <label className="rundeepInventoryCategorySelect"><span className="sr-only">{t('category')}: {shortName(entry.moduleName)}</span><select value={entry.categoryId} disabled={pendingEntry !== undefined || pendingCategory} onChange={event => { void assign(entry.entryId, event.currentTarget.value) }}>
                {plugins.categories.map(option => <option key={option.id} value={option.id}>{option.name}</option>)}
              </select></label>
              <span className="rundeepInventoryState">{t(entry.enabled ? 'enabled' : 'disabled')}</span>
              <button type="button" className="rundeepInventorySwitch" role="switch" aria-checked={entry.enabled} aria-label={`${t(entry.enabled ? 'disable' : 'enable')}: ${shortName(entry.moduleName)}`} disabled={pendingEntry !== undefined || pendingCategory} onClick={() => { void toggle(entry) }}><span /></button>
            </li>)}
          </ul>}
        </div> : null}
      </section>
    })}
  </section>
}

function installStyles() {
  const style = document.createElement('style')
  style.textContent = `
.rundeepInventory{display:flex;flex-direction:column;gap:12px}.rundeepInventoryTools{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,.8fr);gap:12px;align-items:end}
.rundeepInventorySearch,.rundeepInventoryCategoryForm label{display:flex;gap:10px;align-items:center}.rundeepInventorySearch span,.rundeepInventoryCategoryForm label span{font-size:13px;color:var(--dsw-alias-label-tertiary)}.rundeepInventorySearch input,.rundeepInventoryCategoryForm input,.rundeepInventoryCategorySelect select{min-width:0;flex:1;border:1px solid var(--dsw-alias-border-l2);border-radius:8px;background:var(--dsw-alias-bg-layer-3);color:inherit;padding:8px 10px}.rundeepInventoryCategoryForm{display:flex;gap:8px;align-items:end}.rundeepInventoryCategoryForm label{flex:1}.rundeepInventoryCategoryForm button,.rundeepInventoryDeleteCategory{border:1px solid var(--dsw-alias-border-l2);border-radius:7px;background:var(--dsw-alias-bg-layer-3);color:inherit;padding:7px 10px;cursor:pointer}
.rundeepInventoryGroup{overflow:hidden;border:1px solid var(--dsw-alias-border-l2);border-radius:10px;background:var(--dsw-alias-bg-layer-2)}.rundeepInventoryGroupHeading{display:flex;align-items:center}.rundeepInventoryGroupHeader{display:flex;min-width:0;flex:1;align-items:center;gap:8px;border:0;background:transparent;color:inherit;padding:12px 14px;text-align:left;font:inherit;font-size:13px;font-weight:650;cursor:pointer}.rundeepInventoryGroupHeader small{margin-left:auto;color:var(--dsw-alias-label-tertiary);font-weight:500}.rundeepInventoryGroupHeader i{font-style:normal;transition:transform .16s ease}.rundeepInventoryGroupHeader[aria-expanded="true"] i{transform:rotate(180deg)}.rundeepInventoryDeleteCategory{margin-right:10px;padding:5px 8px;font-size:12px;color:var(--dsw-alias-label-tertiary)}
.rundeepInventoryGroupBody{border-top:1px solid var(--dsw-alias-border-l2);padding:8px}.rundeepInventoryGroupBody>p{margin:0;padding:12px;color:var(--dsw-alias-label-tertiary);font-size:13px}.rundeepInventoryGroup ul{display:flex;flex-direction:column;gap:4px;margin:0;padding:0;list-style:none}.rundeepInventoryGroup li{display:grid;grid-template-columns:minmax(150px,1fr) minmax(140px,220px) auto 38px;align-items:center;gap:10px;padding:9px 10px;border-radius:8px;background:var(--dsw-alias-bg-layer-3);font-size:13px}.rundeepInventoryName{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}.rundeepInventoryState{color:var(--dsw-alias-label-tertiary);font-size:12px;white-space:nowrap}.rundeepInventoryCategorySelect{display:flex}.rundeepInventoryCategorySelect select{padding:5px 7px;font-size:12px}
.rundeepInventorySwitch{position:relative;width:36px;height:20px;border:0;border-radius:999px;background:var(--dsw-alias-fill-tertiary);padding:2px;cursor:pointer;transition:background .16s ease}.rundeepInventorySwitch span{display:block;width:16px;height:16px;border-radius:50%;background:white;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .16s ease}.rundeepInventorySwitch[aria-checked="true"]{background:#4f7cff}.rundeepInventorySwitch[aria-checked="true"] span{transform:translateX(16px)}.rundeepInventorySwitch:disabled,.rundeepInventoryCategoryForm button:disabled,.rundeepInventoryDeleteCategory:disabled{cursor:not-allowed;opacity:.38}
.rundeepInventoryRestart{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid rgba(79,124,255,.42);border-radius:9px;background:rgba(79,124,255,.10);padding:10px 12px;font-size:13px}.rundeepInventoryRestart button,.rundeepInventoryStatus button{border:0;border-radius:7px;background:#4f7cff;color:white;padding:6px 10px;cursor:pointer}.rundeepInventoryRestart button:disabled{opacity:.6}.rundeepInventoryError{margin:0;color:var(--dsw-alias-label-error,#e45c5c);font-size:13px}.rundeepInventorySaving,.rundeepInventoryStatus{margin:0;color:var(--dsw-alias-label-tertiary);font-size:13px}@media(max-width:760px){.rundeepInventoryTools{grid-template-columns:1fr}.rundeepInventoryGroup li{grid-template-columns:minmax(0,1fr) 38px}.rundeepInventoryCategorySelect,.rundeepInventoryState{display:none}}
`
  document.head.append(style)
  return () => style.remove()
}

/** Replace the upstream tab body with the launcher-managed effective Loader view. */
export function applyRunDeepPluginInventory(ctx: ClientContext) {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'rundeep: plugin inventory dictionaries')
  ctx.effect(installStyles, 'rundeep: plugin inventory styles')
  const api = createDesktopSettingsApi()
  ctx.slots.inject('settings.plugins.tab', () => ctx.slots.register({
    name: 'settings.plugins.tab', id: 'all', order: 10, label: () => ctx.locale.bind(NS)('tab'), locale: NS, inject: () => ({ api }),
  }, HarnessPluginInventory))
}
