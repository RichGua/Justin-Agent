import { join } from "node:path";
import { homedir } from "node:os";
import { PluginRuntime } from "@justin/host";
import { apply as storage } from "@justin/plugin-storage";
import { apply as codex } from "@justin/plugin-codex";
import { apply as webGateway } from "@justin/plugin-web-gateway";

const dataHome = process.env.JUSTIN_HOME ?? join(homedir(), ".justin-next");
const runtime = new PluginRuntime();
await runtime.load("storage", storage, { path: join(dataHome, "justin.sqlite") });
await runtime.load("codex", codex, { home: join(dataHome, "codex") });
await runtime.load("webGateway", webGateway, { staticDir: join(process.cwd(), "apps/web/dist") });
const gateway = runtime.ctx.justin.services.get("webGateway") as { url: string };
console.log("Justin Agent:", gateway.url);
const stop = async () => { await runtime.close(); process.exit(0); };
process.once("SIGINT", stop); process.once("SIGTERM", stop);
