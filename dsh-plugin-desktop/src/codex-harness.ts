/**
 * Alternative primary AgentFactory backed by the official OpenAI Codex SDK.
 *
 * The lifecycle and inbox scheduling follow DeepSeek Harness's MIT-licensed
 * ReactLoopAgent/AgentLoop contracts, while Codex owns its native thread,
 * tools, sandbox, approvals, and turn execution through @openai/codex-sdk.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { extname, join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import {
  Inbox,
  agentEvents,
  emitAgentEvent,
  type Agent,
  type AgentEventDispatch,
  type AgentFactory,
  type AgentHandle,
  type AgentOptions,
  type AgentSetup,
  type AgentSetupCommit,
  type AgentStatus,
  type CancelOptions,
  type CreateAgentOptions,
  type InboxTarget,
  type ModelSelection,
  type PreStepDecision,
  type ResumeAgentOptions,
  type SessionStartSource,
} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-agent-default-model'
import {
  CallId,
  createAssistantMessage,
  createToolResultMessage,
  errorChain,
  type ContentBlock,
  type ReplayEnvelope,
  type StreamChunk,
  type TokenUsage,
} from '@deepseek-ai/dsh-llm'
import { createScope, type Scope } from '@deepseek-ai/dsh-scope'
import {
  SessionPreparation,
  type AgentCancelCause,
  type Session,
  type SessionId,
  type TurnEndReason,
  type UserMessage,
} from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-session-persistence'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import type {} from '@deepseek-ai/dsh-credentials'
import z from '@deepseek-ai/schemastery'
import {
  Codex,
  type ApprovalMode,
  type Input,
  type ModelReasoningEffort,
  type SandboxMode,
  type ThreadEvent,
  type ThreadItem,
  type ThreadOptions,
  type TurnOptions,
  type Usage,
  type UserInput,
  type WebSearchMode,
} from '@openai/codex-sdk'

export const name = 'codex-harness'
export const inject = ['agents', 'sessions', 'attachments']

declare module '@deepseek-ai/dsh-session' {
  interface SessionEventMap {
    /** Legacy Codex thread identity written by Desktop 2.0.2 and earlier. */
    'codex/thread': { threadId: string }
    /** Legacy lossless Codex diagnostic item written by Desktop 2.0.2 and earlier. */
    'codex/item': { turn: number; step: number; item: ThreadItem }
  }
}

const CODEX_REPLAY_KIND = 'dsh-plugin-desktop/codex-thread'
const CODEX_REPLAY_VERSION = 1

/** DeepSeek Responses API endpoint reused from the DSH base model. */
const DEEPSEEK_API_BASE = 'https://api.deepseek.com'
/** Default credential reference of the DSH base DeepSeek provider. */
const DEEPSEEK_API_KEY_ENV = 'DEEPSEEK_API_KEY'

/** Reasoning-effort ids the Codex CLI accepts; a base model id outside this set is ignored. */
const CODEX_REASONING_EFFORTS = new Set<string>(['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'])

interface CodexReplayResponse {
  readonly kind: typeof CODEX_REPLAY_KIND
  readonly version: typeof CODEX_REPLAY_VERSION
  readonly threadId: string
}

function codexReplayState(threadId: string): ReplayEnvelope {
  return {
    response: {
      kind: CODEX_REPLAY_KIND,
      version: CODEX_REPLAY_VERSION,
      threadId,
    } satisfies CodexReplayResponse,
  }
}

function codexThreadIdFromReplayState(value: unknown): string | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  const response = (value as { response?: unknown }).response
  if (typeof response !== 'object' || response === null) return undefined
  const candidate = response as Partial<CodexReplayResponse>
  return candidate.kind === CODEX_REPLAY_KIND
    && candidate.version === CODEX_REPLAY_VERSION
    && typeof candidate.threadId === 'string'
    && candidate.threadId.length > 0
    ? candidate.threadId
    : undefined
}

/** Resolve the newest native thread identity from the standard replay seam or a legacy event. */
function persistedCodexThreadId(events: readonly Session['events'][number][]): string | undefined {
  for (let index = events.length - 1; index >= 0; index--) {
    const event = events[index]
    if (event?.type === 'codex/thread') {
      if (event.data.threadId.length > 0) return event.data.threadId
    } else if (event?.type === 'assistant/message') {
      const source = event.data.message.source
      if (source.kind === 'model' && source.provider === 'codex') {
        const threadId = codexThreadIdFromReplayState(source.replayState)
        if (threadId !== undefined) return threadId
      }
    } else if (event?.type === 'assistant/chunk' && event.data.chunk.type === 'finish') {
      const threadId = codexThreadIdFromReplayState(event.data.chunk.replayState)
      if (threadId !== undefined) return threadId
    }
  }
  return undefined
}

/** Loader-owned defaults for native Codex execution. */
export interface Config {
  model?: string
  sandboxMode?: SandboxMode
  approvalPolicy?: ApprovalMode
  skipGitRepoCheck?: boolean
  modelReasoningEffort?: ModelReasoningEffort
  networkAccessEnabled?: boolean
  webSearchMode?: WebSearchMode
  /** OpenAI-compatible endpoint used instead of the Codex default; any base URL is valid. */
  baseUrl?: string
  /** Credential reference (environment-variable name) for the endpoint API key. */
  apiKeyRef?: string
  /** Reuse the DSH base model (endpoint, key, model, reasoning) when no explicit values are set. */
  useBaseModel?: boolean
}

export const Config: z<Config> = z.object({
  model: z.string(),
  sandboxMode: z.union(['read-only', 'workspace-write', 'danger-full-access'] as const)
    .default('workspace-write'),
  approvalPolicy: z.union(['never', 'on-request', 'on-failure', 'untrusted'] as const)
    .default('on-request'),
  skipGitRepoCheck: z.boolean().default(true),
  modelReasoningEffort: z.union(['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const),
  networkAccessEnabled: z.boolean(),
  webSearchMode: z.union(['disabled', 'cached', 'live'] as const),
  baseUrl: z.string(),
  apiKeyRef: z.string(),
  useBaseModel: z.boolean().default(true),
})

/** Narrow official SDK faces used by the adapter and replaceable in tests. */
export interface CodexThreadLike {
  readonly id: string | null
  runStreamed(input: Input, options?: TurnOptions): Promise<{ events: AsyncGenerator<ThreadEvent> }>
}

export interface CodexClientLike {
  startThread(options?: ThreadOptions): CodexThreadLike
  resumeThread(id: string, options?: ThreadOptions): CodexThreadLike
}

type Phase =
  | { kind: 'idle'; lastTurn: number }
  | { kind: 'maintenance'; abort: AbortController; lastTurn: number; wakeRequested: boolean }
  | { kind: 'running'; abort: AbortController; turn: number; step: number; wakeRequested: boolean }

interface PreparedAgent {
  agent: CodexHarnessAgent
  signal: AbortSignal
  publish(source: SessionStartSource): AgentHandle
  dispose(): Promise<void>
}

interface LiveBlock {
  readonly id: string
  readonly index: number
  readonly type: 'text' | 'reasoning'
  text: string
  ended: boolean
}

function mediaExtension(mediaType: string, nameHint?: string): string {
  const hinted = nameHint === undefined ? '' : extname(nameHint).toLowerCase()
  if (['.png', '.jpg', '.jpeg', '.webp', '.gif'].includes(hinted)) return hinted
  if (mediaType === 'image/jpeg') return '.jpg'
  if (mediaType === 'image/webp') return '.webp'
  if (mediaType === 'image/gif') return '.gif'
  return '.png'
}

function nestedText(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type === 'text' || block.type === 'reasoning') {
      parts.push(block.text)
    } else if (block.type === 'tool-result') {
      parts.push(nestedText(block.content))
    } else if (block.type === 'tool-call') {
      parts.push(`[tool call ${block.name}: ${block.arguments}]`)
    }
  }
  return parts.join('\n')
}

function usageFromCodex(usage: Usage): TokenUsage {
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    cacheReadTokens: usage.cached_input_tokens,
    cacheWriteTokens: usage.cache_write_input_tokens,
    reasoningTokens: usage.reasoning_output_tokens,
  }
}

function throwAbort(signal: AbortSignal): never {
  signal.throwIfAborted()
  throw new Error('Codex turn aborted')
}

/** Serialize the request payload of one Codex tool item for the DSH tool card. */
function toolArgumentsJson(item: ThreadItem): string {
  switch (item.type) {
    case 'command_execution': return JSON.stringify({ command: item.command })
    case 'file_change': return JSON.stringify({ changes: item.changes })
    case 'mcp_tool_call': return JSON.stringify({ server: item.server, tool: item.tool, arguments: item.arguments })
    case 'web_search': return JSON.stringify({ query: item.query })
    default: return '{}'
  }
}

/** Project one Codex tool item's terminal output as the DSH tool-result text. */
function toolOutputText(item: ThreadItem): string {
  switch (item.type) {
    case 'command_execution': {
      const output = item.aggregated_output.trim()
      return output.length === 0 ? `exit ${item.exit_code ?? 'unknown'}` : output
    }
    case 'file_change': return item.changes.map(change => `${change.kind} ${change.path}`).join('\n')
    case 'mcp_tool_call': {
      if (item.result !== undefined) return JSON.stringify(item.result.content)
      return item.error?.message ?? ''
    }
    case 'web_search': return ''
    default: return ''
  }
}

/** One live DSH Agent whose model/tool driver is an official Codex thread. */
export class CodexHarnessAgent implements Agent {
  readonly inbox: Inbox
  readonly scope: Scope
  readonly ctx: Context
  readonly options: AgentOptions
  private readonly dispatch: AgentEventDispatch
  private phase: Phase
  private activityDone: Promise<void> = Promise.resolve()
  private thread?: CodexThreadLike
  private threadId: string | undefined

  constructor(
    private readonly hostCtx: Context,
    public readonly id: SessionId,
    requestedOptions: AgentOptions,
    public readonly session: Session,
    private readonly codexPromise: Promise<CodexClientLike>,
    private readonly config: Config,
    private readonly baseModel?: ModelSelection,
  ) {
    const requestedCodexModel = requestedOptions.provider === 'codex' ? requestedOptions.model : undefined
    const model = requestedCodexModel ?? config.model ?? baseModel?.model
    this.options = { provider: 'codex', ...(model === undefined ? {} : { model }) }
    this.dispatch = agentEvents(hostCtx, this)
    this.inbox = new Inbox(session, {
      inserted: message => { this.dispatch.emit('agent/inbox/inserted', { message }) },
      discarded: message => { this.dispatch.emit('agent/inbox/discarded', { message }) },
      claimed: (message, turn) => { this.dispatch.emit('agent/inbox/claimed', { message, turn }) },
    })
    const lastTurn = session.events.findLast(event => event.type === 'turn/start')?.data.turn ?? 0
    this.phase = { kind: 'idle', lastTurn }
    this.scope = createScope(hostCtx, this)
    this.ctx = this.scope.ctx.extend({ agent: this })
    this.threadId = persistedCodexThreadId(session.events)
  }

  get status(): AgentStatus {
    return this.phase.kind === 'running' ? 'running' : 'idle'
  }

  private setPhase(next: Phase): void {
    const previous = this.status
    this.phase = next
    if (this.status !== previous) this.dispatch.emit('agent/status', { status: this.status })
  }

  send(message: UserMessage, target: InboxTarget, wakeup: boolean): void {
    const wakingAfterAbort = wakeup && this.phase.kind !== 'idle' && this.phase.abort.signal.aborted
    this.inbox.splice(wakingAfterAbort ? 'next-turn' : target, Infinity, 0, [message])
    if (wakeup) this.wakeDriver(wakingAfterAbort)
  }

  followup(message: UserMessage): void { this.send(message, 'next-turn', true) }
  steer(message: UserMessage): void { this.send(message, 'next-step', true) }
  inject(message: UserMessage): void { this.send(message, 'next-step', false) }

  cancel(cause: AgentCancelCause, options: CancelOptions = {}): void {
    if (!options.keepInbox) {
      this.inbox.clear()
      if (this.phase.kind !== 'idle') this.phase.wakeRequested = false
    }
    if (this.phase.kind !== 'idle') this.phase.abort.abort(cause)
  }

  runMaintenance<T>(job: (signal: AbortSignal) => Promise<T>): Promise<T> {
    if (this.phase.kind !== 'idle') throw new Error(`agent "${this.id}" already has active work`)
    const done = Promise.withResolvers<void>()
    const maintenance: Phase = {
      kind: 'maintenance', abort: new AbortController(), lastTurn: this.phase.lastTurn, wakeRequested: false,
    }
    this.setPhase(maintenance)
    this.activityDone = done.promise
    return (async () => {
      try {
        return await job(maintenance.abort.signal)
      } finally {
        this.setPhase({ kind: 'idle', lastTurn: maintenance.lastTurn })
        if (maintenance.wakeRequested && this.inbox.hasPending) this.wakeDriver()
        done.resolve()
      }
    })()
  }

  async whenIdle(): Promise<void> {
    let activity: Promise<void>
    do {
      await (activity = this.activityDone)
    } while (activity !== this.activityDone)
  }

  private wakeDriver(wakeAfterAbort = false): void {
    if (this.phase.kind !== 'idle') {
      const reason = this.phase.abort.signal.reason as AgentCancelCause | undefined
      if (reason?.kind !== 'disposed' && (this.phase.kind === 'maintenance' || wakeAfterAbort)) {
        this.phase.wakeRequested = true
      }
      return
    }
    const done = Promise.withResolvers<void>()
    this.activityDone = done.promise
    this.setPhase({
      kind: 'running', abort: new AbortController(), turn: this.phase.lastTurn, step: 0, wakeRequested: false,
    })
    this.hostCtx.agents.withInitiator(this, () => this.kick()).then(done.resolve, done.reject)
  }

  private async kick(): Promise<void> {
    try {
      while (await this.turn()) {}
    } catch {
      // Each failure is already projected into agent/error and turn/end.
    } finally {
      if (this.phase.kind === 'running') {
        const { turn, wakeRequested } = this.phase
        this.setPhase({ kind: 'idle', lastTurn: turn })
        if (wakeRequested && this.inbox.hasPending) this.wakeDriver()
      }
    }
  }

  private async preStep(target: InboxTarget, turn: number, step: number): Promise<PreStepDecision> {
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": pre-step outside running phase`)
    const signal = this.phase.abort.signal
    const claimed = this.inbox.claim(target, turn)
    const decision = await this.dispatch.waterfall(
      'agent/pre-step', { messages: claimed, turn, step, signal },
      () => Promise.resolve<PreStepDecision>({ kind: 'enter', messages: claimed }),
    )
    signal.throwIfAborted()
    return decision
  }

  private reportError(error: unknown): void {
    const turn = this.phase.kind === 'running' ? this.phase.turn : this.phase.lastTurn
    const step = this.phase.kind === 'running' ? this.phase.step : 0
    this.dispatch.emit('agent/error', { turn, step, error })
  }

  private async turn(): Promise<boolean> {
    if (this.phase.kind !== 'running') throw new Error(`agent "${this.id}": turn without driver reservation`)
    const phase = this.phase
    const { signal } = phase.abort
    signal.throwIfAborted()
    const turn = phase.turn + 1
    this.session.append('turn/start', { turn })
    phase.turn = turn
    let reason: TurnEndReason = { kind: 'completed' }
    let target: InboxTarget = 'next-turn'
    try {
      while (true) {
        const step = phase.step + 1
        const decision = await this.preStep(target, turn, step)
        if (decision.kind === 'reject') {
          reason = { kind: 'blocked' }
          break
        }
        if (phase.step === 0 && decision.messages.length === 0) break
        if (phase.step > 0 && decision.messages.length === 0) break
        signal.throwIfAborted()
        this.session.append('step/start', { turn, step })
        phase.step = step
        try {
          for (const message of decision.messages) {
            this.session.append('user/message', message, { surfaceOp: 'append' })
          }
          await this.runCodexStep(decision.messages, turn, step, signal)
        } finally {
          this.session.append('step/end', { turn, step })
        }
        signal.throwIfAborted()
        if (this.inbox.nextStep.length === 0) {
          await this.dispatch.serial('agent/turn-stopping', { turn, signal })
          signal.throwIfAborted()
        }
        if (this.inbox.nextStep.length === 0) break
        target = 'next-step'
      }
    } catch (error: unknown) {
      if (signal.aborted) {
        reason = { kind: 'aborted', reason: signal.reason as AgentCancelCause }
      } else {
        reason = { kind: 'error', error: { message: errorChain(error), code: 'CODEX_SDK' } }
      }
      this.reportError(error)
    } finally {
      this.session.append('turn/end', { turn, reason })
    }
    if (!this.inbox.hasPending) return false
    phase.abort = new AbortController()
    phase.wakeRequested = false
    phase.step = 0
    return true
  }

  private threadOptions(): ThreadOptions {
    const baseEffort = this.baseModel?.reasoningEffort
    const reasoningEffort = this.config.modelReasoningEffort
      ?? (baseEffort !== undefined && CODEX_REASONING_EFFORTS.has(baseEffort)
        ? baseEffort as ModelReasoningEffort
        : undefined)
    return {
      ...(this.options.model === undefined ? {} : { model: this.options.model }),
      ...(this.session.header.cwd === undefined ? {} : { workingDirectory: this.session.header.cwd }),
      ...(this.config.sandboxMode === undefined ? {} : { sandboxMode: this.config.sandboxMode }),
      ...(this.config.approvalPolicy === undefined ? {} : { approvalPolicy: this.config.approvalPolicy }),
      ...(this.config.skipGitRepoCheck === undefined ? {} : { skipGitRepoCheck: this.config.skipGitRepoCheck }),
      ...(reasoningEffort === undefined ? {} : { modelReasoningEffort: reasoningEffort }),
      ...(this.config.networkAccessEnabled === undefined ? {} : { networkAccessEnabled: this.config.networkAccessEnabled }),
      ...(this.config.webSearchMode === undefined ? {} : { webSearchMode: this.config.webSearchMode }),
    }
  }

  private async getThread(): Promise<CodexThreadLike> {
    if (this.thread !== undefined) return this.thread
    const codex = await this.codexPromise
    this.thread = this.threadId === undefined
      ? codex.startThread(this.threadOptions())
      : codex.resumeThread(this.threadId, this.threadOptions())
    return this.thread
  }

  private async codexInput(messages: readonly UserMessage[], signal: AbortSignal): Promise<{
    input: Input
    cleanup(): Promise<void>
  }> {
    const input: UserInput[] = []
    let directory: string | undefined
    let imageIndex = 0
    for (const message of messages) {
      for (const block of message.content) {
        if (block.type === 'text' || block.type === 'reasoning') {
          input.push({ type: 'text', text: block.text })
        } else if (block.type === 'image') {
          directory ??= await mkdtemp(join(tmpdir(), 'rundeep-codex-'))
          signal.throwIfAborted()
          const stored = await this.hostCtx.attachments.readImage(block.attachment, signal)
          const path = join(directory, `image-${String(imageIndex++)}${mediaExtension(stored.ref.mediaType, stored.ref.name)}`)
          await writeFile(path, stored.data, { signal })
          input.push({ type: 'local_image', path })
        } else if (block.type === 'tool-result') {
          input.push({ type: 'text', text: nestedText(block.content) })
        } else if (block.type === 'tool-call') {
          input.push({ type: 'text', text: `[tool call ${block.name}: ${block.arguments}]` })
        }
      }
    }
    return {
      input,
      cleanup: async () => {
        if (directory !== undefined) await rm(directory, { recursive: true, force: true })
      },
    }
  }

  /** Project one completed Codex tool item into the DSH tool-call/result pair. */
  private projectToolItem(item: ThreadItem, turn: number, step: number): void {
    if (item.type === 'agent_message' || item.type === 'reasoning'
      || item.type === 'todo_list' || item.type === 'error') return
    const callId = CallId(item.id)
    const failed = item.type === 'command_execution' || item.type === 'file_change'
      ? item.status === 'failed'
      : item.type === 'mcp_tool_call'
        ? item.status === 'failed'
        : false
    this.session.append('tool/call', {
      turn,
      step,
      callId,
      name: item.type,
      arguments: toolArgumentsJson(item),
    })
    this.session.append('tool/result', {
      turn,
      step,
      message: createToolResultMessage({
        callId,
        content: [{ type: 'text', text: toolOutputText(item) }],
        isError: failed,
      }),
    }, { surfaceOp: 'append' })
  }

  private async runCodexStep(
    messages: readonly UserMessage[],
    turn: number,
    step: number,
    signal: AbortSignal,
  ): Promise<void> {
    const prepared = await this.codexInput(messages, signal)
    const blocks: LiveBlock[] = []
    const byId = new Map<string, LiveBlock>()
    const chunkSeqs: number[] = []
    let usage: TokenUsage | undefined
    let completed = false
    const appendChunk = (chunk: StreamChunk): void => {
      chunkSeqs.push(this.session.append('assistant/chunk', { turn, step, chunk }).seq)
    }
    const updateBlock = (item: ThreadItem, end: boolean): void => {
      if (item.type !== 'agent_message' && item.type !== 'reasoning') return
      let block = byId.get(item.id)
      if (block === undefined) {
        block = { id: item.id, index: blocks.length, type: item.type === 'reasoning' ? 'reasoning' : 'text', text: '', ended: false }
        blocks.push(block)
        byId.set(item.id, block)
        appendChunk({ type: 'block-start', index: block.index, blockType: block.type })
      }
      if (item.text.startsWith(block.text)) {
        const delta = item.text.slice(block.text.length)
        if (delta.length > 0) {
          appendChunk(block.type === 'reasoning'
            ? { type: 'reasoning-delta', index: block.index, text: delta }
            : { type: 'text-delta', index: block.index, text: delta })
        }
      }
      block.text = item.text
      if (end && !block.ended) {
        block.ended = true
        appendChunk({ type: 'block-end', index: block.index, block: { type: block.type, text: block.text } })
      }
    }
    try {
      const { events } = await (await this.getThread()).runStreamed(prepared.input, { signal })
      for await (const event of events) {
        signal.throwIfAborted()
        if (event.type === 'thread.started') {
          if (this.threadId !== undefined && this.threadId !== event.thread_id) {
            throw new Error(`Codex resumed unexpected thread ${event.thread_id}`)
          }
          if (this.threadId === undefined) {
            this.threadId = event.thread_id
          }
        } else if (event.type === 'item.started' || event.type === 'item.updated') {
          updateBlock(event.item, false)
          if (event.item.type === 'todo_list') {
            this.session.append('todo/write', {
              todos: event.item.items.map(item => ({ content: item.text, status: item.completed ? 'completed' : 'pending' })),
            })
          }
        } else if (event.type === 'item.completed') {
          updateBlock(event.item, true)
          this.projectToolItem(event.item, turn, step)
          if (event.item.type === 'todo_list') {
            this.session.append('todo/write', {
              todos: event.item.items.map(item => ({ content: item.text, status: item.completed ? 'completed' : 'pending' })),
            })
          }
        } else if (event.type === 'turn.completed') {
          usage = usageFromCodex(event.usage)
          completed = true
        } else if (event.type === 'turn.failed') {
          throw new Error(event.error.message)
        } else if (event.type === 'error') {
          throw new Error(event.message)
        }
      }
      if (!completed) throw new Error('Codex event stream ended without turn.completed')
    } catch (error: unknown) {
      for (const block of blocks) {
        if (!block.ended) {
          block.ended = true
          appendChunk({ type: 'block-end', index: block.index, block: { type: block.type, text: block.text } })
        }
      }
      const replayState = this.threadId === undefined ? undefined : codexReplayState(this.threadId)
      appendChunk({
        type: 'finish',
        reason: signal.aborted
          ? { kind: 'aborted', failure: { message: errorChain(error), code: 'CODEX_SDK' } }
          : { kind: 'error', failure: { message: errorChain(error), code: 'CODEX_SDK' } },
        ...(replayState === undefined ? {} : { replayState }),
      })
      if (blocks.some(block => block.text.length > 0)) {
        this.session.append('assistant/message', {
          turn,
          step,
          message: createAssistantMessage({
            content: blocks.map(block => ({ type: block.type, text: block.text })),
            source: {
              provider: 'codex',
              model: this.options.model ?? 'codex-default',
              ...(replayState === undefined ? {} : { replayState }),
            },
          }),
          interrupted: true,
        }, { surfaceOp: 'append', sourceEventSeqs: chunkSeqs })
      }
      if (signal.aborted) throwAbort(signal)
      throw error
    } finally {
      await prepared.cleanup()
    }
    for (const block of blocks) {
      if (!block.ended) {
        block.ended = true
        appendChunk({ type: 'block-end', index: block.index, block: { type: block.type, text: block.text } })
      }
    }
    if (usage !== undefined) appendChunk({ type: 'usage', usage })
    const replayState = this.threadId === undefined ? undefined : codexReplayState(this.threadId)
    appendChunk({
      type: 'finish',
      reason: { kind: 'stop' },
      ...(replayState === undefined ? {} : { replayState }),
    })
    this.session.append('assistant/message', {
      turn,
      step,
      message: createAssistantMessage({
        content: blocks.map(block => ({ type: block.type, text: block.text })),
        source: {
          provider: 'codex',
          model: this.options.model ?? 'codex-default',
          ...(replayState === undefined ? {} : { replayState }),
        },
      }),
      ...(usage === undefined ? {} : { usage }),
    }, { surfaceOp: 'append', sourceEventSeqs: chunkSeqs })
  }
}

async function awaitSetup(
  setup: AgentSetup | undefined,
  agent: CodexHarnessAgent,
  signal: AbortSignal,
): Promise<AgentSetupCommit | void> {
  signal.throwIfAborted()
  const operation = Promise.resolve(setup?.(agent.ctx))
  const aborted = Promise.withResolvers<never>()
  const onAbort = (): void => { aborted.reject(signal.reason) }
  signal.addEventListener('abort', onAbort, { once: true })
  try {
    return await Promise.race([operation, aborted.promise])
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}

/** DSH AgentFactory implementation selected in place of dsh-agent-loop. */
export class CodexHarnessFactory implements AgentFactory {
  private readonly teardown = new AbortController()
  private readonly live = new Set<() => Promise<void>>()
  private readonly codexPromise: Promise<CodexClientLike>
  private readonly baseModel: ModelSelection | undefined

  constructor(
    private readonly ctx: Context,
    private readonly config: Config,
    codex?: CodexClientLike,
  ) {
    this.baseModel = config.useBaseModel === false
      ? undefined
      : ctx.get('agentDefaultModel')?.currentSelection()
    this.codexPromise = codex === undefined ? this.buildCodex(ctx, config) : Promise.resolve(codex)
  }

  /** Construct the official SDK client, resolving the endpoint key per factory. */
  private async buildCodex(ctx: Context, config: Config): Promise<CodexClientLike> {
    const base = this.baseModel
    const apiKeyRef = config.apiKeyRef ?? (base?.provider === 'deepseek' ? DEEPSEEK_API_KEY_ENV : undefined)
    let apiKey: string | undefined
    if (apiKeyRef !== undefined) {
      const credentials = ctx.get('credentials')
      if (credentials !== undefined) {
        const resolved = await credentials.resolve(credentialRef(apiKeyRef))
        apiKey = resolved?.value
      }
    }
    const baseUrl = config.baseUrl ?? (base?.provider === 'deepseek' ? DEEPSEEK_API_BASE : undefined)
    return new Codex({
      ...(baseUrl === undefined ? {} : { baseUrl }),
      ...(apiKey === undefined ? {} : { apiKey }),
    })
  }

  private prepare(
    ownerCtx: Context,
    id: SessionId,
    options: AgentOptions,
    session: Session,
    callerSignal?: AbortSignal,
  ): PreparedAgent {
    ownerCtx.fiber.assertActive()
    const lifecycleAbort = new AbortController()
    const forwardCallerAbort = (): void => { lifecycleAbort.abort(callerSignal?.reason) }
    const forwardFactoryAbort = (): void => { lifecycleAbort.abort(this.teardown.signal.reason) }
    callerSignal?.addEventListener('abort', forwardCallerAbort, { once: true })
    this.teardown.signal.addEventListener('abort', forwardFactoryAbort, { once: true })
    if (callerSignal?.aborted) forwardCallerAbort()
    if (this.teardown.signal.aborted) forwardFactoryAbort()

    const agent = new CodexHarnessAgent(
      this.ctx,
      id,
      options,
      session,
      this.codexPromise,
      this.config,
      this.baseModel,
    )
    let detachSession: (() => void) | undefined
    let detachAgent: (() => void) | undefined
    let disposing: Promise<void> | undefined
    let ownerTriggered = false
    let unfollowOwner: () => Promise<void> | void = () => {}
    const dispose = (): Promise<void> => (disposing ??= (async () => {
      lifecycleAbort.abort(new Error(`agent "${id}" lifecycle disposed`))
      callerSignal?.removeEventListener('abort', forwardCallerAbort)
      this.teardown.signal.removeEventListener('abort', forwardFactoryAbort)
      agent.cancel({ kind: 'disposed' })
      await agent.whenIdle()
      await agent.scope.dispose()
      detachAgent?.()
      detachSession?.()
      this.live.delete(dispose)
      if (!ownerTriggered) await unfollowOwner()
    })())
    this.live.add(dispose)
    unfollowOwner = ownerCtx.effect(() => () => {
      // External disposal unregisters this owner effect from inside the same
      // shared teardown. Returning that in-flight Promise here would await
      // itself; this is the same convergence guard used by AgentLoop.
      if (disposing !== undefined) return
      ownerTriggered = true
      return dispose()
    }, `codexHarness.lifecycle(${id})`)

    const assertLive = (): void => {
      if (!lifecycleAbort.signal.aborted) return
      throw lifecycleAbort.signal.reason instanceof Error
        ? lifecycleAbort.signal.reason
        : new Error(`agent "${id}" lifecycle aborted`, { cause: lifecycleAbort.signal.reason })
    }
    return {
      agent,
      signal: lifecycleAbort.signal,
      publish: (source) => {
        assertLive()
        detachSession = agent.ctx.sessions.enter(session)
        detachAgent = this.ctx.agents.enter(agent, ownerCtx.agent)
        agent.ctx.sessions.announce(session)
        assertLive()
        this.ctx.agents.announce(agent)
        assertLive()
        emitAgentEvent(this.ctx, agent, 'agent/session-start', { source })
        assertLive()
        return { agent, dispose }
      },
      dispose,
    }
  }

  private async setupAndPublish(
    ownerCtx: Context,
    id: SessionId,
    preparation: SessionPreparation,
    options: AgentOptions,
    setup: AgentSetup | undefined,
    signal: AbortSignal | undefined,
    source: SessionStartSource,
  ): Promise<AgentHandle> {
    const prepared = this.prepare(ownerCtx, id, options, preparation.session, signal)
    try {
      const commit = await awaitSetup(setup, prepared.agent, prepared.signal)
      commit?.commit()
      return prepared.publish(source)
    } catch (error: unknown) {
      await prepared.dispose()
      throw error
    } finally {
      preparation[Symbol.dispose]()
    }
  }

  async createAgent(ownerCtx: Context, options: CreateAgentOptions): Promise<AgentHandle> {
    const preparation = SessionPreparation.create(this.ctx.sessions.prepare(options.sessionId, {
      ...(options.seed === undefined ? {} : { seed: options.seed }),
      ...(options.meta === undefined ? {} : { meta: options.meta }),
    }))
    return this.setupAndPublish(
      ownerCtx,
      options.sessionId,
      preparation,
      options.agentOptions ?? {},
      options.setup,
      options.signal,
      'startup',
    )
  }

  async resume(ownerCtx: Context, options: ResumeAgentOptions): Promise<AgentHandle> {
    const persistence = this.ctx.get('sessionPersistence')
    if (persistence === undefined) {
      throw new Error('cannot resume: session persistence is not configured')
    }
    const preparation = await persistence.prepare(options.resumeSessionId, options.signal)
    return this.setupAndPublish(
      ownerCtx,
      options.resumeSessionId,
      preparation,
      options.agentOptions ?? {},
      options.setup,
      options.signal,
      'resume',
    )
  }

  async dispose(): Promise<void> {
    if (!this.teardown.signal.aborted) this.teardown.abort(new Error('Codex Harness is not active'))
    await Promise.all([...this.live].map(dispose => dispose()))
  }
}

/** Register the official-SDK AgentFactory in the one primary-agent slot. */
export function apply(ctx: Context, config: Config): void {
  const factory = new CodexHarnessFactory(ctx, config)
  ctx.effect(() => ctx.agents.setFactory(factory), 'codexHarness.setFactory()')
  ctx.effect(() => () => factory.dispose(), 'codexHarness.transactions()')
}
