import { z } from "zod";

export const JUSTIN_API_VERSION = 1 as const;
export const permissionSchema = z.enum([
  "filesystem:read", "filesystem:write", "network", "process", "notifications", "ui",
]);
export type JustinPermission = z.infer<typeof permissionSchema>;

export const contributionSchema = z.object({
  slot: z.enum(["leftSidebar", "rightInspector", "settings", "commandPalette"]),
  entry: z.string().min(1),
});

export const manifestSchema = z.object({
  apiVersion: z.literal(JUSTIN_API_VERSION),
  host: z.string().optional(),
  client: z.string().optional(),
  bundle: z.string().optional(),
  configSchema: z.record(z.string(), z.unknown()).optional(),
  permissions: z.array(permissionSchema).default([]),
  dependencies: z.record(z.string(), z.string()).default({}),
  contributions: z.array(contributionSchema).default([]),
});
export type JustinManifest = z.infer<typeof manifestSchema>;

export const rpcRequestSchema = z.object({
  jsonrpc: z.literal("2.0"),
  id: z.string().or(z.number()),
  method: z.string().regex(/^[a-z][a-z0-9]*(\.[a-zA-Z0-9]+)+$/),
  params: z.unknown().optional(),
});
export type JustinRpcRequest = z.infer<typeof rpcRequestSchema>;

export type Disposable = () => void | Promise<void>;
export type PluginApply<Context, Config = unknown> = (ctx: Context, config: Config) =>
  void | Disposable | Promise<void | Disposable>;
