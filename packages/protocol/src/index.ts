import { z } from "zod";
import { rpcRequestSchema } from "@justin/sdk";

export const protocolVersion = "justin.v1" as const;
export const helloSchema = z.object({ protocol: z.literal(protocolVersion), locale: z.enum(["zh-CN", "en"]).default("zh-CN") });
export const notificationSchema = z.object({ method: z.string().regex(/^[a-z][a-z0-9]*(\.[a-zA-Z0-9]+)+$/), params: z.unknown() });
export { rpcRequestSchema };
