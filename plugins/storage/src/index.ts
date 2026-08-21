import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { provide, type JustinContext } from "@justin/host";

export interface SettingsStore {
  get(key: string): unknown | undefined;
  set(key: string, value: unknown): void;
}
export function apply(ctx: JustinContext, config: { path: string }): () => void {
  mkdirSync(dirname(config.path), { recursive: true });
  const database = new DatabaseSync(config.path);
  database.exec("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
  const service: SettingsStore = {
    get(key) {
      const row = database.prepare("SELECT value FROM settings WHERE key = ?").get(key) as { value: string } | undefined;
      return row && JSON.parse(row.value);
    },
    set(key, value) { database.prepare("INSERT OR REPLACE INTO settings VALUES (?, ?)").run(key, JSON.stringify(value)); },
  };
  const revoke = provide(ctx, "settings", service);
  return () => { revoke(); database.close(); };
}
