import { completeSimple } from "@earendil-works/pi-ai/compat";
import type { Api, AssistantMessage, Model, UserMessage } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { CruiseControlConfig, Instructions } from "./config";
import { type Classification, type ClassificationRequest, isLevel } from "./types";

/** Tool input is truncated before it reaches the classifier; verdicts hinge on shape, not bulk. */
const MAX_INPUT_CHARS = 4000;
const MAX_PROMPT_CHARS = 600;
const MAX_REASON_CHARS = 200;
const MAX_REASON_WORDS = 20;

const OUTPUT_CONTRACT = `You are a tool-use gate for the pi coding agent. Classify one pending tool call.

Reply with ONE JSON object and nothing else. No prose, no markdown fences:
{"risk":"low|medium|high","intent":"low|medium|high","reason":"<one sentence, under 20 words>"}

risk - potential impact of running this tool call:
  low    = read-only, trivially reversible, or confined to scratch state
  medium = local mutation inside the workspace that a human could undo
  high   = destructive, irreversible, privileged, remote, or outside the workspace

intent - how clearly the user asked for this, judged from the recent prompts:
  low    = unrelated to anything the user asked for
  medium = a plausible step toward the user's request
  high   = the user explicitly asked for this action

reason - the justification. When risk is high, name the specific danger so the agent
can propose a safer call instead of guessing.`;

/**
 * `retryable` marks failures that another attempt could plausibly fix — endpoint
 * errors, per-attempt timeouts, and unparseable replies. Configuration faults such as
 * a missing model or missing credentials are not retryable: retrying only delays the
 * fallback verdict the caller is already going to take.
 */
export class ClassifierError extends Error {
  constructor(
    message: string,
    readonly retryable = false,
  ) {
    super(message);
    this.name = "ClassifierError";
  }
}

/**
 * Models that answered one classification attempt with a temperature rejection,
 * remembered for the process lifetime. Only the first call against such a model
 * pays for the failed request; later calls omit the parameter from the start.
 */
const temperatureRejected = new Set<string>();

function modelKey(model: Model<Api>): string {
  return `${model.provider}/${model.id}`;
}

/** Read `compat.supportsTemperature` without fighting the conditional `compat` union on `Model<Api>`. */
function compatSupportsTemperature(model: Model<Api>): boolean | undefined {
  const compat = model.compat as { supportsTemperature?: boolean } | undefined;
  return compat?.supportsTemperature;
}

/**
 * Whether `temperature: 0` should be sent for this model.
 *
 * Not every backend accepts it:
 * - the ChatGPT Codex backend (`openai-codex-responses`) rejects the parameter
 *   outright with HTTP 400, and nothing in its model metadata says so;
 * - Claude Opus 4.7+ deprecated the parameter, and pi flags those models
 *   `compat.supportsTemperature: false`. That flag exists only on the anthropic
 *   compat type, so this check fires just for anthropic-messages models —
 *   defense-in-depth on top of pi-ai's own gating. Claude behind an
 *   OpenAI-compatible proxy carries no flag and is covered by the reactive
 *   fallback below instead.
 *
 * Anything else gets the parameter once — if the endpoint complains, the call is
 * retried without it and the model is remembered.
 */
function shouldSendTemperature(model: Model<Api>): boolean {
  if (temperatureRejected.has(modelKey(model))) return false;
  if (model.api === "openai-codex-responses") return false;
  if (compatSupportsTemperature(model) === false) return false;
  return true;
}

/** Every known temperature rejection names the parameter in its error text. */
function isTemperatureError(error: string): boolean {
  return /temperature/i.test(error);
}

/**
 * Ask the configured model to rate one tool call.
 *
 * Rejects with `ClassifierError` when no model resolves, auth is missing, the call
 * aborts or times out, or the reply is not a usable verdict. The gate turns that
 * into the configured `onError` outcome.
 */
export async function classify(
  request: ClassificationRequest,
  config: CruiseControlConfig,
  ctx: ExtensionContext,
  signal: AbortSignal,
): Promise<Classification> {
  const model = resolveModel(config, ctx);
  if (!model) {
    throw new ClassifierError(
      config.model ? `classifier model "${config.model}" is not available` : "no classifier model selected",
    );
  }

  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok) throw new ClassifierError(auth.error);

  const message: UserMessage = {
    role: "user",
    content: [{ type: "text", text: buildPayload(request) }],
    timestamp: Date.now(),
  };

  // `temperature: 0` favors consistent verdicts, but a model that rejects the
  // parameter rejects it on every attempt, so a plain retry loop can never
  // succeed against it — without this fallback every classification on such a
  // backend fails and the gate runs on its `onError` policy alone.
  const requestOnce = (temperature?: number) =>
    completeSimple(
      model,
      { systemPrompt: buildSystemPrompt(config.instructions), messages: [message] },
      {
        apiKey: auth.apiKey,
        headers: auth.headers,
        env: auth.env,
        reasoning: config.reasoning,
        temperature,
        signal,
      },
    );

  const withTemperature = shouldSendTemperature(model);
  let outcome = await attempt(requestOnce, withTemperature);
  if (!outcome.ok && withTemperature && !signal.aborted && isTemperatureError(outcome.error)) {
    temperatureRejected.add(modelKey(model));
    outcome = await attempt(requestOnce, false);
  }
  if (!outcome.ok) throw new ClassifierError(outcome.error, true);

  const response = outcome.message;
  if (response.stopReason === "aborted") throw new ClassifierError("classification aborted", true);

  const text = response.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("\n");

  const parsed = parseClassification(text);
  if (!parsed) throw new ClassifierError("classifier returned no usable verdict", true);
  return parsed;
}

/** Resolve `provider/modelId` against the registry, falling back to the session model. */
export function resolveModel(config: CruiseControlConfig, ctx: ExtensionContext): Model<Api> | undefined {
  if (!config.model) return ctx.model;

  const separator = config.model.indexOf("/");
  if (separator <= 0) return ctx.model;

  const provider = config.model.slice(0, separator);
  const modelId = config.model.slice(separator + 1);
  return ctx.modelRegistry.find(provider, modelId) ?? undefined;
}

/**
 * One classification attempt. Endpoint faults surface two ways depending on the
 * API — a thrown error, or a response with an `error` stop reason — so both are
 * folded into a failure string; everything else comes back as the response.
 */
async function attempt(
  request: (temperature?: number) => Promise<AssistantMessage>,
  withTemperature: boolean,
): Promise<{ ok: true; message: AssistantMessage } | { ok: false; error: string }> {
  try {
    const message = await request(withTemperature ? 0 : undefined);
    if (message.stopReason === "error") {
      return { ok: false, error: message.errorMessage ?? "classifier request failed" };
    }
    return { ok: true, message };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

export function buildSystemPrompt(instructions: Instructions): string {
  const sections: string[] = [OUTPUT_CONTRACT];
  const append = (title: string, lines: string[]) => {
    if (lines.length > 0) sections.push(`${title}\n${lines.map((line) => `- ${line}`).join("\n")}`);
  };

  append("Background:", instructions.background);
  append("Allow (normally low risk):", instructions.allow);
  append("Conditional (judge case by case):", instructions.conditional);
  append("Deny (treat as high risk):", instructions.deny);

  return sections.join("\n\n");
}

function buildPayload(request: ClassificationRequest): string {
  const payload = {
    tool: request.toolName,
    input: truncate(safeJson(request.input), MAX_INPUT_CHARS),
    cwd: request.cwd,
    recent_user_prompts: request.recentPrompts.map((prompt) => truncate(prompt, MAX_PROMPT_CHARS)),
  };
  return `Tool call to classify:\n${JSON.stringify(payload, null, 2)}`;
}

/**
 * Pull a verdict out of the reply. Models wrap JSON in fences or commentary often
 * enough that scanning for the outermost object is worth the leniency.
 */
export function parseClassification(text: string): Classification | undefined {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return undefined;

  let parsed: unknown;
  try {
    parsed = JSON.parse(text.slice(start, end + 1));
  } catch {
    return undefined;
  }

  if (!parsed || typeof parsed !== "object") return undefined;
  const candidate = parsed as Record<string, unknown>;
  if (!isLevel(candidate.risk) || !isLevel(candidate.intent)) return undefined;

  const reason = typeof candidate.reason === "string" ? candidate.reason.trim() : "";
  return {
    risk: candidate.risk,
    intent: candidate.intent,
    reason: clampReason(reason || "no reason given"),
  };
}

function clampReason(reason: string): string {
  const collapsed = reason.replace(/\s+/g, " ").trim();
  const words = collapsed.split(" ");
  const clipped = words.length > MAX_REASON_WORDS ? `${words.slice(0, MAX_REASON_WORDS).join(" ")}…` : collapsed;
  return truncate(clipped, MAX_REASON_CHARS);
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

function truncate(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}… (truncated)`;
}
