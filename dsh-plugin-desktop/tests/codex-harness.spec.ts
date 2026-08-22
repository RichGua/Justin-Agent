import { Context } from '@deepseek-ai/cordis'
import AgentRegistry from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId, type SessionEvent } from '@deepseek-ai/dsh-session'
import type { Input, ThreadEvent, ThreadOptions } from '@openai/codex-sdk'
import { describe, expect, it, vi } from 'vitest'
import {
  CodexHarnessFactory,
  Config,
  type CodexClientLike,
  type CodexThreadLike,
} from '../src/codex-harness.ts'

function eventStream(events: readonly ThreadEvent[]): AsyncGenerator<ThreadEvent> {
  return (async function * () { for (const event of events) yield event })()
}

const SUCCESS_EVENTS: readonly ThreadEvent[] = [
  { type: 'thread.started', thread_id: 'codex-thread-1' },
  { type: 'turn.started' },
  { type: 'item.started', item: { id: 'answer-1', type: 'agent_message', text: '' } },
  { type: 'item.updated', item: { id: 'answer-1', type: 'agent_message', text: 'Hello' } },
  { type: 'item.completed', item: { id: 'answer-1', type: 'agent_message', text: 'Hello from Codex' } },
  {
    type: 'turn.completed',
    usage: {
      input_tokens: 11,
      cached_input_tokens: 3,
      cache_write_input_tokens: 2,
      output_tokens: 5,
      reasoning_output_tokens: 1,
    },
  },
]

class FakeThread implements CodexThreadLike {
  readonly id: string | null = null
  readonly inputs: Input[] = []
  constructor(private readonly events: readonly ThreadEvent[]) {}
  async runStreamed(input: Input): Promise<{ events: AsyncGenerator<ThreadEvent> }> {
    this.inputs.push(input)
    return { events: eventStream(this.events) }
  }
}

class FakeCodex implements CodexClientLike {
  readonly started: ThreadOptions[] = []
  readonly resumed: { id: string; options: ThreadOptions | undefined }[] = []
  readonly threads: FakeThread[] = []

  startThread(options?: ThreadOptions): CodexThreadLike {
    this.started.push(options ?? {})
    const thread = new FakeThread(SUCCESS_EVENTS)
    this.threads.push(thread)
    return thread
  }

  resumeThread(id: string, options?: ThreadOptions): CodexThreadLike {
    this.resumed.push({ id, options })
    const thread = new FakeThread(SUCCESS_EVENTS)
    this.threads.push(thread)
    return thread
  }
}

async function harness(codex = new FakeCodex()) {
  const ctx = new Context()
  await ctx.plugin(SessionStore)
  await ctx.plugin(AgentRegistry)
  let runtime!: Context
  await ctx.inject(['agents', 'sessions'], (child: Context) => { runtime = child })
  const factory = new CodexHarnessFactory(runtime, Config({}), codex)
  runtime.effect(() => runtime.agents.setFactory(factory))
  return { ctx, factory, codex }
}

function prompt(text: string) {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

describe('Codex Harness AgentFactory', () => {
  it('uses the official SDK thread and projects its stream into the DSH session', async () => {
    const test = await harness()
    const handle = await test.ctx.agents.create({
      sessionId: SessionId('codex-session-1'),
      meta: { cwd: 'C:\\workspace' },
      agentOptions: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
    })

    handle.agent.followup(prompt('Use the official SDK'))
    await handle.agent.whenIdle()

    expect(handle.agent.options).toEqual({ provider: 'codex' })
    expect(test.codex.started).toEqual([expect.objectContaining({
      workingDirectory: 'C:\\workspace',
      sandboxMode: 'workspace-write',
      approvalPolicy: 'never',
      skipGitRepoCheck: true,
    })])
    expect(test.codex.threads[0]?.inputs).toEqual([[
      { type: 'text', text: 'Use the official SDK' },
    ]])
    expect(handle.agent.session.events.find(event => event.type === 'codex/thread')?.data)
      .toEqual({ threadId: 'codex-thread-1' })
    const assistant = handle.agent.session.events.findLast(event => event.type === 'assistant/message')
    expect(assistant?.data.message.content).toEqual([{ type: 'text', text: 'Hello from Codex' }])
    expect(assistant?.data.usage).toEqual({
      inputTokens: 11,
      outputTokens: 5,
      cacheReadTokens: 3,
      cacheWriteTokens: 2,
      reasoningTokens: 1,
    })
    expect(handle.agent.session.events.findLast(event => event.type === 'turn/end')?.data.reason)
      .toEqual({ kind: 'completed' })

    await handle.dispose()
    await test.factory.dispose()
    await test.ctx.fiber.dispose()
  })

  it('resumes the exact persisted Codex thread instead of starting a replacement', async () => {
    const test = await harness()
    const id = SessionId('codex-session-resume')
    const first = await test.ctx.agents.create({ sessionId: id })
    first.agent.followup(prompt('First turn'))
    await first.agent.whenIdle()
    const seed = structuredClone(first.agent.session.events) as SessionEvent[]
    await first.dispose()

    const second = await test.ctx.agents.create({ sessionId: id, seed })
    second.agent.followup(prompt('Second turn'))
    await second.agent.whenIdle()

    expect(test.codex.started).toHaveLength(1)
    expect(test.codex.resumed).toEqual([expect.objectContaining({ id: 'codex-thread-1' })])
    expect(second.agent.session.events.filter(event => event.type === 'codex/thread')).toHaveLength(1)

    await second.dispose()
    await test.factory.dispose()
    await test.ctx.fiber.dispose()
  })

  it('records SDK failures as structured DSH turn errors', async () => {
    const codex = new FakeCodex()
    vi.spyOn(codex, 'startThread').mockImplementation((options) => {
      codex.started.push(options ?? {})
      const thread = new FakeThread([
        { type: 'thread.started', thread_id: 'failed-thread' },
        { type: 'turn.started' },
        { type: 'turn.failed', error: { message: 'native Codex failure' } },
      ])
      codex.threads.push(thread)
      return thread
    })
    const test = await harness(codex)
    const handle = await test.ctx.agents.create({ sessionId: SessionId('codex-session-error') })

    handle.agent.followup(prompt('Fail this turn'))
    await handle.agent.whenIdle()

    expect(handle.agent.session.events.findLast(event => event.type === 'turn/end')?.data.reason)
      .toEqual({ kind: 'error', error: { message: 'native Codex failure', code: 'CODEX_SDK' } })

    await handle.dispose()
    await test.factory.dispose()
    await test.ctx.fiber.dispose()
  })

  it('accepts an independent endpoint and resolves its key through the credentials seam', async () => {
    const ctx = new Context()
    const resolve = vi.fn(async () => ({ value: 'sk-independent', source: 'env' }))
    ctx.provide('credentials', { resolve } as never)
    const factory = new CodexHarnessFactory(
      ctx,
      Config({ baseUrl: 'https://api.example.com/v1', apiKeyRef: 'CODEX_API_KEY' }),
    )

    await new Promise(resolveImmediate => setImmediate(resolveImmediate))
    expect(resolve).toHaveBeenCalledWith(expect.objectContaining({}))
    await factory.dispose()
    await ctx.fiber.dispose()
  })

  it('builds the default official client when no endpoint or credentials service is configured', async () => {
    const ctx = new Context()
    const factory = new CodexHarnessFactory(ctx, Config({}))

    await new Promise(resolveImmediate => setImmediate(resolveImmediate))
    await factory.dispose()
    await ctx.fiber.dispose()
  })
})
