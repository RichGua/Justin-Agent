import { useEffect, useMemo, useState } from 'react'
import type { PluginInventorySnapshot } from '@deepseek-ai/dsh-api-remotes/client'
import type { PluginInventoryEntry } from '@deepseek-ai/dsh-host-plugin-inventory/types'
import type { ClientContext } from '@deepseek-ai/dsh-client-runtime/client'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type {} from '@deepseek-ai/dsh-client-ui-settings-plugins/client'
import type { PropsLocale, PropsRuntime, InjectFace } from '@deepseek-ai/dsh-client-ui-slots'

const NS = 'rundeep.pluginInventory'
type HarnessFamily = 'deepseek' | 'codex' | 'extension'

const zh = {
  tab: '按 Harness 分类',
  loading: '正在读取已加载的插件…',
  error: '暂时无法读取插件列表。',
  retry: '重试',
  search: '搜索插件',
  deepseek: 'DeepSeek 原生 Harness',
  codex: 'Codex Harness',
  extension: 'RunDeep 扩展 / 未来 Harness',
  noDeepseek: '没有已加载的 DeepSeek 原生插件。',
  noCodex: '尚未安装或启用 Codex Harness 插件。',
  noExtension: '没有其他扩展插件；Claude Harness 将显示在这里。',
  enabled: '已启用',
  disabled: '已停用',
} as const
type LocaleKey = keyof typeof zh
const en: Record<LocaleKey, string> = {
  tab: 'By Harness', loading: 'Reading loaded plugins…', error: 'The plugin list is temporarily unavailable.', retry: 'Retry', search: 'Search plugins',
  deepseek: 'Native DeepSeek Harness', codex: 'Codex Harness', extension: 'RunDeep extensions / future Harnesses',
  noDeepseek: 'No native DeepSeek plugins are loaded.', noCodex: 'No Codex Harness plugins are installed or enabled.', noExtension: 'No other extensions are loaded. Claude Harness will appear here.',
  enabled: 'Enabled', disabled: 'Disabled',
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap { 'rundeep.pluginInventory': LocaleKey }
}

interface Injected { list: () => Promise<PluginInventorySnapshot> }
type Props = PropsRuntime<'settings.plugins.tab'> & PropsLocale<typeof NS> & InjectFace<Injected>

function familyOf(moduleName: string): HarnessFamily {
  if (moduleName === '@justin-agent/dsh-harness-codex' || /(?:^|[-/])codex(?:$|[-/])/iu.test(moduleName)) return 'codex'
  if (moduleName.startsWith('@deepseek-ai/')) return 'deepseek'
  return 'extension'
}

function shortName(moduleName: string) {
  return moduleName.replace(/^@deepseek-ai\/dsh-/u, '').replace(/^@justin-agent\/dsh-harness-/u, '')
}

function HarnessPluginInventory({ list, t }: Props) {
  const [snapshot, setSnapshot] = useState<PluginInventorySnapshot>()
  const [failed, setFailed] = useState(false)
  const [query, setQuery] = useState('')
  const load = () => {
    setFailed(false)
    void list().then(setSnapshot, () => setFailed(true))
  }
  useEffect(load, [list])
  const groups = useMemo(() => {
    const result: Record<HarnessFamily, PluginInventoryEntry[]> = { deepseek: [], codex: [], extension: [] }
    const needle = query.trim().toLocaleLowerCase()
    for (const entry of snapshot?.entries ?? []) {
      if (needle && !entry.moduleName.toLocaleLowerCase().includes(needle)) continue
      result[familyOf(entry.moduleName)].push(entry)
    }
    return result
  }, [query, snapshot])
  if (snapshot === undefined && !failed) return <p className="rundeepInventoryStatus">{t('loading')}</p>
  if (failed) return <div className="rundeepInventoryStatus"><p role="alert">{t('error')}</p><button type="button" onClick={load}>{t('retry')}</button></div>
  return <section className="rundeepInventory">
    <label className="rundeepInventorySearch"><span>{t('search')}</span><input type="search" value={query} onChange={event => setQuery(event.currentTarget.value)} placeholder={t('search')} /></label>
    {(['deepseek', 'codex', 'extension'] as const).map(family => <section className="rundeepInventoryGroup" key={family}>
      <h3>{t(family)}</h3>
      {groups[family].length === 0 ? <p>{t(family === 'deepseek' ? 'noDeepseek' : family === 'codex' ? 'noCodex' : 'noExtension')}</p> : <ul>
        {groups[family].map(entry => <li key={entry.entryId}>
          <span title={entry.moduleName}>{shortName(entry.moduleName)}</span>
          <small>{entry.enabled ? t('enabled') : t('disabled')}</small>
        </li>)}
      </ul>}
    </section>)}
  </section>
}

function installStyles() {
  const style = document.createElement('style')
  style.textContent = '.rundeepInventory{display:flex;flex-direction:column;gap:18px}.rundeepInventorySearch{display:flex;gap:10px;align-items:center}.rundeepInventorySearch span{font-size:13px;color:var(--dsw-alias-label-tertiary)}.rundeepInventorySearch input{min-width:220px;flex:1;border:1px solid var(--dsw-alias-border-l2);border-radius:8px;background:var(--dsw-alias-bg-layer-3);color:inherit;padding:8px 10px}.rundeepInventoryGroup h3{margin:0 0 8px;font-size:13px}.rundeepInventoryGroup>p{margin:0;padding:12px;border:1px dashed var(--dsw-alias-border-l2);border-radius:8px;color:var(--dsw-alias-label-tertiary);font-size:13px}.rundeepInventoryGroup ul{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:0;padding:0;list-style:none}.rundeepInventoryGroup li{display:flex;justify-content:space-between;gap:8px;padding:11px 12px;border:1px solid var(--dsw-alias-border-l2);border-radius:9px;background:var(--dsw-alias-bg-layer-3);font-size:13px}.rundeepInventoryGroup li span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}.rundeepInventoryGroup li small{color:var(--dsw-alias-label-tertiary);white-space:nowrap}.rundeepInventoryStatus{color:var(--dsw-alias-label-tertiary)}'
  document.head.append(style)
  return () => style.remove()
}

/** Own the visible inventory tab while retaining the upstream read-only Remote contract. */
export function applyRunDeepPluginInventory(ctx: ClientContext) {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'rundeep: plugin inventory dictionaries')
  ctx.effect(installStyles, 'rundeep: plugin inventory styles')
  const list = async () => {
    const result = await ctx.remote.pluginInventory.list()
    if (!result.ok) throw new Error(result.error.message)
    return result.value
  }
  ctx.slots.inject('settings.plugins.tab', () => ctx.slots.register({
    name: 'settings.plugins.tab', id: 'all', order: 10, label: () => ctx.locale.bind(NS)('tab'), locale: NS, inject: () => ({ list }),
  }, HarnessPluginInventory))
}
