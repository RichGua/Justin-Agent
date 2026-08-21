import Fastify from "fastify";
import cookie from "@fastify/cookie";
import websocket from "@fastify/websocket";
import fastifyStatic from "@fastify/static";
import { resolve } from "node:path";
import { nanoid } from "nanoid";
import { rpcRequestSchema } from "@justin/protocol";
import { provide, type CodexAppServer, type JustinContext } from "@justin/host";

export interface WebGateway { url: string; close(): Promise<void>; }
export async function apply(ctx: JustinContext, config: { port?: number; staticDir?: string }): Promise<() => Promise<void>> {
  const app = Fastify({ logger: false, trustProxy: false });
  await app.register(cookie);
  await app.register(websocket);
  if (config.staticDir) await app.register(fastifyStatic, { root: resolve(config.staticDir), wildcard: false });
  const token = nanoid(32);
  const sessions = new Set<string>();
  const origin = (request: { headers: { origin?: string | undefined } }) => request.headers.origin?.startsWith("http://127.0.0.1:") === true;
  app.get("/health", async () => ({ ok: true }));
  app.get("/login/:token", async (request, reply) => {
    if ((request.params as { token: string }).token !== token) return reply.code(404).send();
    const session = nanoid(32); sessions.add(session);
    reply.setCookie("justin_session", session, { httpOnly: true, sameSite: "strict", secure: false, path: "/" });
    return reply.redirect("/", 302);
  });
  const clients = new Set<{ send(payload: string): void; close(code?: number, reason?: string): void }>();
  const codex = ctx.justin.services.get("codex") as CodexAppServer | undefined;
  if (codex) codex.on("notification", (message) => {
    const payload = JSON.stringify({ jsonrpc: "2.0", method: "codex.event", params: message });
    for (const client of clients) client.send(payload);
  });
  app.get("/rpc", { websocket: true }, (socket, request) => {
    const session = request.cookies.justin_session;
    if (!session || !sessions.has(session) || !origin(request)) return socket.close(1008, "unauthorized");
    clients.add(socket);
    socket.on("close", () => clients.delete(socket));
    socket.on("message", async (raw: Buffer) => {
      let value: unknown;
      try { value = JSON.parse(raw.toString()); } catch { return socket.send(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } })); }
      const parsed = rpcRequestSchema.safeParse(value);
      if (!parsed.success) return socket.send(JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32600, message: "Invalid request" } }));
      const { id, method, params } = parsed.data;
      if (method === "system.ping") socket.send(JSON.stringify({ jsonrpc: "2.0", id, result: { protocol: "justin.v1" } }));
      else if (!codex) socket.send(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32000, message: "Codex is unavailable" } }));
      else {
        const methodMap: Record<string, string> = {
          "threads.list": "thread/list", "threads.create": "thread/start",
          "threads.resume": "thread/resume", "threads.send": "turn/start",
          "turns.interrupt": "turn/interrupt",
        };
        const codexMethod = methodMap[method];
        if (!codexMethod) return socket.send(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found" } }));
        try {
          const result = await codex.request(codexMethod, params);
          socket.send(JSON.stringify({ jsonrpc: "2.0", id, result }));
        } catch (error) {
          socket.send(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32000, message: error instanceof Error ? error.message : "Codex request failed" } }));
        }
      }
    });
  });
  await app.listen({ host: "127.0.0.1", port: config.port ?? 0 });
  const address = app.server.address();
  if (!address || typeof address === "string") throw new Error("Expected TCP listener");
  const service: WebGateway = { url: `http://127.0.0.1:${address.port}/login/${token}`.replace(/\\`/g, "`"), close: () => app.close() };
  const revoke = provide(ctx, "webGateway", service);
  return async () => { revoke(); await service.close(); };
}

