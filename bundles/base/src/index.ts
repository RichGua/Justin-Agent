export const baseBundle = [
  { id: "storage", package: "@justin/plugin-storage", config: {} },
  { id: "codex", package: "@justin/plugin-codex", config: {}, dependencies: ["storage"] },
] as const;
