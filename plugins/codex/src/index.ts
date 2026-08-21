import { CodexAppServer, provide, type JustinContext } from "@justin/host";
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";

export async function apply(ctx: JustinContext, config: { binary?: string; home: string }): Promise<() => void> {
  mkdirSync(config.home, { recursive: true });
  const require = createRequire(import.meta.url);
  const configuredBinary = config.binary ?? process.env.JUSTIN_CODEX_BIN;
  const server = new CodexAppServer(
    configuredBinary ?? process.execPath,
    configuredBinary ? undefined : [require.resolve("@openai/codex/bin/codex.js"), "app-server", "--listen", "stdio://"],
    {
    ...process.env, CODEX_HOME: config.home,
    },
  );
  server.start();
  server.on("stderr", (line) => process.stderr.write("[codex] " + line));
  await server.initialize({ name: "justin-agent", title: "Justin Agent", version: "0.1.0" });
  const revoke = provide(ctx, "codex", server);
  return () => { revoke(); server.stop(); };
}
