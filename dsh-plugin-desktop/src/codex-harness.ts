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
  type PreStepDecision,
  type ResumeAgentOptions,
  type SessionStartSource,
} from '@deepseek-ai/dsh-agent'
import {
  createAssistantMessage,
  errorChain,
  type ContentBlock,
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
    /** Official Codex thread identity used to continue after a DSH resume. */
    'codex/thread': { threadId: string }
    /** Lossless completed Codex item for diagnostics and future UI adapters. */
    'codex/item': { turn: number; step: number; item: ThreadItem }
  }
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
}

export const Config: z<Config> = z.object({
  model: z.string(),
  sandboxMode: z.union(['read-only', 'workspace-write', 'danger-full-access'] as const)
    .default('workspace-write'),
  approvalPolicy: z.union(['never', 'on-request', 'on-failure', 'untrusted'] as const)
    .default('never'),
  skipGitRepoCheck: z.boolean().default(true),
  modelReasoningEffort: z.union(['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const),
  networkAccessEnabled: z.boolean(),
  webSearchMode: z.union(['disabled', 'cached', 'live'] as const),
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
    private readonly codex: CodexClientLike,
    private readonly config: Config,
  ) {
    const requestedCodexModel = requestedOptions.provider === 'codex' ? requestedOptions.model : undefined
    const model = requestedCodexModel ?? config.model
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
    this.threadId = session.events.findLast(event => event.type === 'codex/thread')?.data.threadId
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
    return {
      ...(this.options.model === undefined ? {} : { model: this.options.model }),
      ...(this.session.header.cwd === undefined ? {} : { workingDirectory: this.session.header.cwd }),
      ...(this.config.sandboxMode === undefined ? {} : { sandboxMode: this.config.sandboxMode }),
      ...(this.config.approvalPolicy === undefined ? {} : { approvalPolicy: this.config.approvalPolicy }),
      ...(this.config.skipGitRepoCheck === undefined ? {} : { skipGitRepoCheck: this.config.skipGitRepoCheck }),
      ...(this.config.modelReasoningEffort === undefined ? {} : { modelReasoningEffort: this.config.modelReasoningEffort }),
      ...(this.config.networkAccessEnabled === undefined ? {} : { networkAccessEnabled: this.config.networkAccessEnabled }),
      ...(this.config.webSearchMode === undefined ? {} : { webSearchMode: this.config.webSearchMode }),
    }
  }

  private getThread(): CodexThreadLike {
    if (this.thread !== undefined) return this.thread
    this.thread = this.threadId === undefined
      ? this.codex.startThread(this.threadOptions())
      : this.codex.resumeThread(this.threadId, this.threadOptions())
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
      const { events } = await this.getThread().runStreamed(prepared.input, { signal })
      for await (const event of events) {
        signal.throwIfAborted()
        if (event.type === 'thread.started') {
          if (this.threadId !== undefined && this.threadId !== event.thread_id) {
            throw new Error(`Codex resumed unexpected thread ${event.thread_id}`)
          }
          if (this.threadId === undefined) {
            this.threadId = event.thread_id
            this.session.append('codex/thread', { threadId: event.thread_id })
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
          this.session.append('codex/item', { turn, step, item: event.item })
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
      if (blocks.some(block => block.text.length > 0)) {
        this.session.append('assistant/message', {
          turn,
          step,
          message: createAssistantMessage({
            content: blocks.map(block => ({ type: block.type, text: block.text })),
            source: { provider: 'codex', model: this.options.model ?? 'codex-default' },
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
    appendChunk({ type: 'finish', reason: { kind: 'stop' } })
    this.session.append('assistant/message', {
      turn,
      step,
      message: createAssistantMessage({
        content: blocks.map(block => ({ type: block.type, text: block.text })),
        source: { provider: 'codex', model: this.options.model ?? 'codex-default' },
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

  constructor(
    private readonly ctx: Context,
    private readonly config: Config,
    private readonly codex: CodexClientLike = new Codex(),
  ) {}

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

    const agent = new CodexHarnessAgent(this.ctx, id, options, session, this.codex, this.config)
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
