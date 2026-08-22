import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { createServer, type IncomingMessage, type Server } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Codex } from '@openai/codex-sdk'
import { afterEach, describe, expect, it } from 'vitest'
import { codexEndpointClientOptions } from '../src/codex-harness.ts'

const roots: string[] = []
const servers: Server[] = []

afterEach(async () => {
  await Promise.all(servers.splice(0).map(closeServer))
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

function readRequest(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = ''
    request.setEncoding('utf8')
    request.on('data', (chunk: string) => { body += chunk })
    request.on('end', () => { resolve(body) })
    request.on('error', reject)
  })
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close(error => {
      if (error !== undefined) reject(error)
      else resolve()
    })
    server.closeAllConnections()
  })
}

function responseEvents(text: string): Record<string, unknown>[] {
  const part = { type: 'output_text', annotations: [], logprobs: [], text }
  const message = {
    id: 'msg_rundeep_fixture',
    type: 'message',
    status: 'completed',
    role: 'assistant',
    content: [part],
  }
  const completed = {
    id: 'resp_rundeep_fixture',
    object: 'response',
    created_at: 1,
    status: 'completed',
    background: false,
    error: null,
    incomplete_details: null,
    instructions: null,
    max_output_tokens: null,
    max_tool_calls: null,
    model: 'fixture-model',
    output: [message],
    parallel_tool_calls: true,
    previous_response_id: null,
    prompt_cache_key: null,
    prompt_cache_retention: null,
    reasoning: { effort: null, summary: null },
    safety_identifier: null,
    service_tier: 'default',
    store: false,
    temperature: null,
    text: { format: { type: 'text' }, verbosity: 'medium' },
    tool_choice: 'auto',
    tools: [],
    top_logprobs: 0,
    top_p: null,
    truncation: 'disabled',
    usage: {
      input_tokens: 1,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 1,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 2,
    },
    user: null,
    metadata: {},
  }
  return [
    { type: 'response.created', response: { ...completed, status: 'in_progress', output: [] } },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: { ...message, status: 'in_progress', content: [] },
    },
    {
      type: 'response.content_part.added',
      item_id: message.id,
      output_index: 0,
      content_index: 0,
      part: { ...part, text: '' },
    },
    {
      type: 'response.output_text.delta',
      item_id: message.id,
      output_index: 0,
      content_index: 0,
      delta: text,
      logprobs: [],
    },
    {
      type: 'response.output_text.done',
      item_id: message.id,
      output_index: 0,
      content_index: 0,
      text,
      logprobs: [],
    },
    {
      type: 'response.content_part.done',
      item_id: message.id,
      output_index: 0,
      content_index: 0,
      part,
    },
    { type: 'response.output_item.done', output_index: 0, item: message },
    { type: 'response.completed', response: completed },
  ]
}

interface Fixture {
  readonly baseUrl: string
  readonly request: Promise<{ path: string | undefined; authorization: string | undefined }>
}

async function startFixture(answer: string): Promise<Fixture> {
  const observed = Promise.withResolvers<{ path: string | undefined; authorization: string | undefined }>()
  const server = createServer((request, response) => {
    void readRequest(request).then(() => {
      observed.resolve({ path: request.url, authorization: request.headers.authorization })
      response.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        'x-request-id': 'req_rundeep_fixture',
      })
      for (const event of responseEvents(answer)) response.write(`data: ${JSON.stringify(event)}\n\n`)
      response.end('data: [DONE]\n\n')
    }).catch((error: unknown) => {
      response.destroy(error instanceof Error ? error : new Error(String(error)))
    })
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject)
      resolve()
    })
  })
  servers.push(server)
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('fixture did not acquire a port')
  return { baseUrl: `http://127.0.0.1:${address.port}`, request: observed.promise }
}

function isolatedEnv(root: string): Record<string, string> {
  const env: Record<string, string> = {
    CODEX_HOME: join(root, 'codex-home'),
    HOME: root,
    USERPROFILE: root,
    HTTP_PROXY: '',
    HTTPS_PROXY: '',
    ALL_PROXY: '',
    NO_PROXY: '127.0.0.1,localhost',
  }
  for (const key of ['ComSpec', 'PATH', 'SystemRoot', 'TEMP', 'TMP']) {
    const value = process.env[key]
    if (value !== undefined) env[key] = value
  }
  return env
}

describe('Codex Harness real provider wiring', () => {
  it('uses the compatible endpoint API key without reading stored OpenAI authentication', async () => {
    const root = await mkdtemp(join(tmpdir(), 'rundeep-codex-provider-'))
    roots.push(root)
    await mkdir(join(root, 'codex-home'))
    const workspace = join(root, 'workspace')
    await mkdir(workspace)
    const fixture = await startFixture('RUNDEEP_PROVIDER_OK')
    const codex = new Codex({
      ...codexEndpointClientOptions(fixture.baseUrl, 'sk-rundeep-fixture'),
      env: isolatedEnv(root),
    })

    const turn = await codex.startThread({
      model: 'fixture-model',
      workingDirectory: workspace,
      skipGitRepoCheck: true,
      sandboxMode: 'read-only',
      approvalPolicy: 'never',
      webSearchMode: 'disabled',
    }).run('Return the fixture response without tools.')

    expect(turn.finalResponse).toBe('RUNDEEP_PROVIDER_OK')
    await expect(fixture.request).resolves.toEqual({
      path: '/responses',
      authorization: 'Bearer sk-rundeep-fixture',
    })
    await expect(readFile(join(root, 'codex-home', 'auth.json'), 'utf8')).rejects.toThrow()
  }, 30_000)
})
