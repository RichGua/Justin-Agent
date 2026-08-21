import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";

type RpcId = number;
type RpcMessage = { id?: RpcId; method?: string; params?: unknown; result?: unknown; error?: { message: string } };
export class CodexAppServer extends EventEmitter {
  #child: ChildProcessWithoutNullStreams | undefined;
  #nextId = 1;
  #pending = new Map<RpcId, { resolve(value: unknown): void; reject(reason: Error): void }>();
  constructor(private readonly command: string, private readonly args = ["app-server", "--listen", "stdio://"], private readonly env: NodeJS.ProcessEnv = process.env) { super(); }
  start(): void {
    if (this.#child) return;
    const child = spawn(this.command, this.args, { stdio: "pipe", env: this.env, windowsHide: true });
    this.#child = child;
    createInterface({ input: child.stdout }).on("line", (line) => this.onLine(line));
    child.stderr.on("data", (chunk) => this.emit("stderr", String(chunk)));
    child.once("exit", (code, signal) => {
      this.#child = undefined;
      const error = new Error("Codex App Server exited (" + (code ?? signal ?? "unknown") + ")");
      for (const { reject } of this.#pending.values()) reject(error);
      this.#pending.clear();
      this.emit("exit", { code, signal });
    });
  }
  async initialize(clientInfo: { name: string; title: string; version: string }): Promise<void> {
    await this.request("initialize", { clientInfo });
    this.notify("initialized", {});
  }
  request(method: string, params?: unknown): Promise<unknown> {
    if (!this.#child) throw new Error("Codex App Server is not running");
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#child?.stdin.write(JSON.stringify({ id, method, params }) + "\n");
    });
  }
  notify(method: string, params?: unknown): void {
    if (!this.#child) throw new Error("Codex App Server is not running");
    this.#child.stdin.write(JSON.stringify({ method, params }) + "\n");
  }
  stop(): void { this.#child?.kill(); }
  private onLine(line: string): void {
    let message: RpcMessage;
    try { message = JSON.parse(line) as RpcMessage; } catch { this.emit("protocolError", line); return; }
    if (message.id !== undefined) {
      const pending = this.#pending.get(message.id);
      if (!pending) return;
      this.#pending.delete(message.id);
      message.error ? pending.reject(new Error(message.error.message)) : pending.resolve(message.result);
    } else if (message.method) this.emit("notification", message);
  }
}


