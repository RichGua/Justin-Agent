/** Compatibility admission for informational session events written by older Desktop builds. */

import { KNOWN_SESSION_EVENT_TYPES } from '@deepseek-ai/dsh-session'

/**
 * Older Codex Harness builds persisted these Desktop-private records without
 * the `ignorable` envelope marker. Core DSH reconstruction never depends on
 * either record: `codex/thread` is adapter continuation metadata and
 * `codex/item` is diagnostic data.
 */
export const LEGACY_CODEX_SESSION_EVENT_TYPES = Object.freeze([
  'codex/thread',
  'codex/item',
] as const)

/**
 * Admit legacy Codex metadata before session persistence serves history.
 *
 * The upstream catalog is exposed as a ReadonlySet because ordinary plugins
 * must not redefine core vocabulary. At runtime it is the shared Set consulted
 * by the persistence coordinator, and upstream currently has no downstream
 * registration API. Keep this narrow shim until such an API or a format
 * migration exists. New Desktop builds do not write these event types.
 */
export function installLegacyCodexSessionEventCompatibility(): void {
  const knownTypes = KNOWN_SESSION_EVENT_TYPES as Set<string>
  for (const type of LEGACY_CODEX_SESSION_EVENT_TYPES) knownTypes.add(type)
}
