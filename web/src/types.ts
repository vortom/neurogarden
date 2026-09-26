// Friendlier names for the generated protocol types (see wire.d.ts, `npm run types`).
import type * as P from "./wire";

export type ServerMessage = P.ServerMessage;
export type ClientMessage = P.ClientMessage;
export type Welcome = P.Welcome;
export type Catalog = P.Catalog;
export type BodyInfo = P.BodyInfo;
export type Joined = P.Joined;
export type Observation = P.Observation;
export type Died = P.Died;
export type WorldMap = P.WorldMap;
export type Frame = P.Frame1;
export type AgentView = P.AgentView;
export type ResourceView = P.ResourceView;
export type OwnerScore = P.OwnerScore;
export type Chronicle = P.Chronicle;
export type ErrorPayload = P.Error;

export const PROTOCOL_VERSION = 1;
