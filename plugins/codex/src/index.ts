import { CodexAppServer, provide, type JustinContext } from "@justin/host";
import { createRequire } from "node:module";

export function apply(ctx: JustinContext, config: { binary?: string; home: string }): () => void {
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
  const revoke = provide(ctx, "codex", server);
  return () => { revoke(); server.stop(); };
}
