import { z } from "zod";

export const bundleLineSchema = z.object({
  id: z.string().min(1),
  package: z.string().min(1),
  config: z.record(z.string(), z.unknown()).default({}),
  dependencies: z.array(z.string()).default([]),
  enabled: z.boolean().default(true),
});
export type BundleLine = z.infer<typeof bundleLineSchema>;
export type ProfileLayer = readonly BundleLine[];

export function composeProfile(...layers: ProfileLayer[]): BundleLine[] {
  const effective = new Map<string, BundleLine>();
  for (const layer of layers) for (const raw of layer) {
    const line = bundleLineSchema.parse(raw);
    for (const dependency of line.dependencies) {
      if (!effective.get(dependency)?.enabled) throw new Error(`${line.id} requires enabled bundle ${dependency}`.replace(/\\`/g, "`"));
    }
    effective.set(line.id, line);
  }
  return [...effective.values()].filter((line) => line.enabled);
}


