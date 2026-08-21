import { describe, expect, it } from "vitest";
import { composeProfile } from "../src/profile.js";

describe("profile composition", () => {
  it("honors later layers and excludes disabled lines", () => {
    expect(composeProfile(
      [{ id: "base", package: "base", config: {}, dependencies: [], enabled: true }],
      [{ id: "base", package: "replacement", config: {}, dependencies: [], enabled: false }],
      [{ id: "web", package: "web", config: {}, dependencies: [], enabled: true }],
    )).toEqual([{ id: "web", package: "web", config: {}, dependencies: [], enabled: true }]);
  });
});
