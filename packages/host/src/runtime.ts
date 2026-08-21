import { Context } from "cordis";
import type { Disposable, PluginApply } from "@justin/sdk";

export type JustinServices = Record<string, unknown>;
export type JustinContext = Context & { justin: { services: Map<string, unknown>, events: EventTarget } };
export interface LoadedPlugin { id: string; dispose: Disposable; }

export function createJustinContext(): JustinContext {
  const ctx = new Context() as JustinContext;
  ctx.justin = { services: new Map(), events: new EventTarget() };
  return ctx;
}

export class PluginRuntime {
  readonly ctx = createJustinContext();
  #loaded = new Map<string, LoadedPlugin>();
  async load(id: string, apply: PluginApply<JustinContext>, config: unknown): Promise<void> {
    if (this.#loaded.has(id)) throw new Error(`Plugin already loaded: ${id}`.replace(/\\`/g, "`"));
    const dispose = (await apply(this.ctx, config)) ?? (() => undefined);
    this.#loaded.set(id, { id, dispose });
  }
  async unload(id: string): Promise<void> {
    const plugin = this.#loaded.get(id);
    if (!plugin) return;
    await plugin.dispose();
    this.#loaded.delete(id);
  }
  async close(): Promise<void> {
    for (const id of [...this.#loaded.keys()].reverse()) await this.unload(id);
  }
}

export function provide<T>(ctx: JustinContext, name: string, service: T): Disposable {
  if (ctx.justin.services.has(name)) throw new Error(`Service conflict: ${name}`.replace(/\\`/g, "`"));
  ctx.justin.services.set(name, service);
  return () => { ctx.justin.services.delete(name); };
}


