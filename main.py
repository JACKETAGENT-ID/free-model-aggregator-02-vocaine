# vocaine-aggregator02: one server, two client wires, fourteen live free models.
#
# A free-model aggregator for make.fun's free-model-aggregator-02. The server
# speaks OpenAI Chat Completions and Anthropic Messages (JSON and SSE), and
# answers every prompt with a real completion from one of the fourteen free
# OpenRouter models in the trial brief, dialled through the grader's gateway.
# Python 3.13 standard library only; nothing is installed before it runs.
#
# VOCAINE ROUTING (this entry's own strategy, per-prompt specialization plus
# deterministic weighted round-robin):
#   * every candidate model carries a live score: rating_milli x reliability,
#     where reliability starts at 1 and halves on each refusal that model
#     suffers during this trial (a refusal is a gateway answer, never a guess);
#   * a streaming prompt takes the highest live score over all free models;
#   * a JSON-mode prompt takes the highest live score over JSON-capable
#     models only; a tools prompt over tools-capable models only;
#   * when several models tie on the top score, their order is a round-robin
#     rotation whose offset is sha256(trial_seed + "|" + prompt_id) mod the
#     size of the tied group -- no wall clock, no unseeded randomness, so a
#     replay of the same brief, requests and gateway answers is byte-identical;
#   * on a refusal the server fails over to the next model in that same
#     rotated order (at most 6 dials a prompt, keeping the gateway-call
#     tie-break low) and reports the refusals it recovered from in the
#     x-makefun-upstream-failures header, so the failover column can see them.
#
# The catalog printed by `python3 main.py --list` is byte for byte the
# catalog registered with the env: the same object, the same encoder, the
# same separators. With no brief on disk the catalog is empty, as the rules
# require outside a trial.
#
# Layout (one file: the trial runs `python3 -P -s main.py`, so no sibling
# module would be importable):
#   part 1  the brief: finding its two documents, the model table, the catalog
#   part 2  routing: live scores, the deterministic rotation, the gateway dial
#   part 3  the wires: OpenAI <-> Anthropic translation, SSE, the HTTP server,
#           register + poll, and start

import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ------------------------------------------------------------------ constants

PROTOCOL_VERSION = 1
POLL_EVERY_SEC = 0.1
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_DIALS_PER_PROMPT = 6
MAX_TRAIL_ROWS = 8
FAILURES_HEADER = "x-makefun-upstream-failures"
DEFAULT_MAX_TOKENS = 1024

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
BRIEF_PATH = os.environ.get("MAKEFUN_PROMPT_FILE", "")
TRIAL_SEED = os.environ.get("MAKEFUN_TRIAL_SEED", "")
GATEWAY_ROOT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

# The contest's schema prefix: the brief's two documents are found by it.
SCHEMA_PREFIX = "free-model-aggregator-02/"
MODELS_DOC_KEYS = ("models", "ratings")
PROMPTS_DOC_KEYS = ("prompts", "manifest")

# The provider's public endpoint, spelled in the catalog; never the gateway.
PUBLIC_BASE = "https://" + "openrouter" + ".ai/api/v1"
PUBLIC_KEY_ENV = "OPENROUTER_API_KEY"

# Fixed timestamps in the model lists: nothing in a reply reads the clock.
LIST_CREATED = 1767225600
LIST_CREATED_AT = "2026-01-01T00:00:00Z"

Json = dict  # one JSON object, as Python reads it


def is_obj(value):
    return isinstance(value, dict)


def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_str(value):
    return value if isinstance(value, str) else ""


def as_int_or_none(value):
    if is_num(value) and value == value and value not in (float("inf"), float("-inf")):
        return value
    return None


def as_int_or_zero(value):
    return value if is_num(value) else 0


# ------------------------------------------------------------------ the brief


def schema_documents(text):
    """Every JSON object in `text` carrying a string `schema`, in order.

    The brief is prose with two JSON documents embedded in it; each document
    is found by scanning for object starts and decoding from there with the
    stdlib decoder, skipping anything that is not a JSON object.
    """
    decoder = json.JSONDecoder()
    found = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            pos = start + 1
            continue
        pos = start + end
        if is_obj(value) and isinstance(value.get("schema"), str):
            found.append(value)
    return found


def pick_document(docs, keys):
    """First document under this contest's prefix holding one of `keys`."""
    for doc in docs:
        schema = as_str(doc.get("schema"))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def parse_model(row):
    """One model row of the brief, normalized; None when it names no model."""
    if not is_obj(row):
        return None
    raw_id = as_str(row.get("id")) or as_str(row.get("model"))
    if raw_id == "":
        return None
    provider = as_str(row.get("provider")) or as_str(row.get("wire")) or "openrouter"
    model_id = raw_id
    if provider == "openrouter" and not raw_id.lower().startswith("openrouter/"):
        model_id = "openrouter/" + raw_id
    rating = as_int_or_none(row.get("rating_milli"))
    context = as_int_or_none(row.get("context_length"))
    return {
        "id": model_id,
        "provider": provider,
        "display_name": as_str(row.get("display_name")) or raw_id,
        "rating_milli": rating if rating is not None else 0,
        "context_length": context,
        "tools": row.get("tools") if isinstance(row.get("tools"), bool) else None,
        "vision": row.get("vision") if isinstance(row.get("vision"), bool) else None,
        "json": row.get("json") if isinstance(row.get("json"), bool) else None,
        "free": row.get("free") is not False,
    }


def parse_prompt(row):
    """One prompt row of the brief, normalized; None when it has no id."""
    if not is_obj(row):
        return None
    prompt_id = as_str(row.get("id"))
    if prompt_id == "":
        return None
    req = row.get("requires") if is_obj(row.get("requires")) else {}
    needs = {}
    if req.get("tools") is True:
        needs["tools"] = True
    if req.get("vision") is True:
        needs["vision"] = True
    if req.get("json") is True:
        needs["json"] = True
    if is_num(req.get("min_context")):
        needs["min_context"] = req["min_context"]
    return {
        "id": prompt_id,
        "api": "anthropic" if row.get("api") == "anthropic" else "openai",
        "stream": row.get("stream") is True,
        "weight": 2 if row.get("weight") == 2 else 1,
        "requires": needs,
    }


def read_brief(path):
    """The brief at `path`: {"models": [...], "prompts": [...], "error": str|None}."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": "cannot read %s: %s" % (path, err)}
    docs = schema_documents(text)
    models_doc = pick_document(docs, MODELS_DOC_KEYS)
    if models_doc is None:
        return {"models": [], "prompts": [], "error": "the brief holds no models document"}
    models = [m for m in (parse_model(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = pick_document(docs, PROMPTS_DOC_KEYS)
    if prompts_doc is None:
        return {"models": models, "prompts": [], "error": "the brief holds no prompts document"}
    prompts = [p for p in (parse_prompt(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {"models": models, "prompts": prompts, "error": None}


BRIEF = read_brief(BRIEF_PATH)


def free_models():
    """The brief's free models, in the brief's order, each once."""
    seen = set()
    out = []
    for model in BRIEF["models"]:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        out.append(model)
    return out


def build_catalog():
    """The catalog document: one provider, exactly the free models.

    base_url is the provider's public endpoint, never the gateway's trial
    address: the gateway origin differs between a live pass and its replay,
    and the catalog is part of the state export a replay compares byte for
    byte.
    """
    models = [
        {"id": m["id"], "free": True, "context_length": m["context_length"]}
        for m in free_models()
    ]
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": PUBLIC_BASE,
                "key_env": PUBLIC_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


CATALOG = build_catalog()
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def catalog_models():
    """The models the catalog serves, as GET /v1/models names them."""
    return [
        {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
        for m in free_models()
    ]


def log(message):
    sys.stderr.write("vocaine-aggregator02: %s\n" % message)
    sys.stderr.flush()


# ------------------------------------------------------------------ routing
# Vocain's strategy: per-prompt specialization over a live score, ties broken
# by a deterministic rotation. Every decision below is a function of the
# brief, the requests and the gateway answers seen so far -- never of the
# wall clock and never of unseeded randomness.

# Refusals suffered per model id, in this trial only. A model's live score is
# rating_milli x (1/2 ** refusals): one refusal halves it, so a rate-limited
# leader falls behind the next candidate at once, and a model that keeps
# answering keeps its rating.
_refusals = {}
_refusals_lock = threading.Lock()


def live_score(model):
    with _refusals_lock:
        fails = _refusals.get(model["id"], 0)
    return model["rating_milli"] * (0.5 ** fails)


def note_refusal(model_id):
    with _refusals_lock:
        _refusals[model_id] = _refusals.get(model_id, 0) + 1


def meets(needs, model):
    """Whether a model's capabilities meet one prompt's requirements.

    An unknown capability never meets a requirement that names it.
    """
    if needs.get("tools") is True and model["tools"] is not True:
        return False
    if needs.get("vision") is True and model["vision"] is not True:
        return False
    if needs.get("json") is True and model["json"] is not True:
        return False
    floor = needs.get("min_context")
    if floor is not None and (model["context_length"] is None or model["context_length"] < floor):
        return False
    return True


def rotation_offset(prompt_id, group_size):
    """The round-robin offset for one prompt: sha256(seed + "|" + id).

    The trial seed is the trial's own; the prompt id is the brief's row for
    the request arriving now. The same prompt in a replay hashes the same way.
    """
    digest = hashlib.sha256(("%s|%s" % (TRIAL_SEED, prompt_id)).encode("utf-8")).hexdigest()
    return int(digest, 16) % group_size if group_size > 0 else 0


def ordered_candidates(needs, prompt_id):
    """The free models that can serve this prompt, best first.

    Models are grouped by live score, highest first; inside a tied group the
    order is the group's id order rotated by the prompt's rotation offset,
    so tied models take turns across prompts instead of the first id always
    winning. Failover walks this same order.
    """
    pool = [m for m in free_models() if meets(needs, m)]
    groups = {}
    for model in pool:
        groups.setdefault(live_score(model), []).append(model)
    ordered = []
    for score in sorted(groups.keys(), reverse=True):
        group = sorted(groups[score], key=lambda m: m["id"])
        if len(group) > 1:
            shift = rotation_offset(prompt_id, len(group))
            group = group[shift:] + group[:shift]
        ordered.extend(group)
    return ordered


# The brief's prompts, in the client's arrival order: the request body never
# names its prompt id, so arrival order is how a request finds its row.
_manifest_at = 0
_manifest_lock = threading.Lock()


def prompt_row_for(body):
    """(prompt_id, requirements) for the request arriving now."""
    global _manifest_at
    with _manifest_lock:
        index = _manifest_at
        _manifest_at += 1
    prompts = BRIEF["prompts"]
    if index < len(prompts):
        row = prompts[index]
        return row["id"], row["requires"]
    return prompt_id_from_body(body), requirements_from_body(body)


def prompt_id_from_body(body):
    """A stable id for a request with no manifest row: a hash of its body."""
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "body-" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def requirements_from_body(body):
    """Requirements read off a request body, when the brief has no row for it."""
    vision = False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_obj(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_obj(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format")
    fmt_type = fmt.get("type") if is_obj(fmt) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt_type in ("json_object", "json_schema"),
    }


# ------------------------------------------------------------------ the gateway


def gateway_chat_url():
    """The gateway's Chat Completions route: $OPENAI_BASE_URL, /v1 appended
    when the base does not already end in it."""
    if GATEWAY_ROOT == "":
        return ""
    if GATEWAY_ROOT.endswith("/v1"):
        return GATEWAY_ROOT + "/chat/completions"
    return GATEWAY_ROOT + "/v1/chat/completions"


class Upstream:
    """One gateway answer: its status and a body read whole or as chunks."""

    def __init__(self, status, response):
        self.status = status
        self._response = response

    @property
    def ok(self):
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self):
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self):
        try:
            self._response.close()
        except Exception:
            pass


def dial(model_id, body):
    """POST one Chat Completions request to the gateway naming `model_id`.

    A refusal comes back as an Upstream with its status and body, exactly as
    the provider sent it; an unreachable gateway raises.
    """
    url = gateway_chat_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    payload = json.dumps({**body, "model": model_id}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": "Bearer %s" % GATEWAY_KEY}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def refusal_code(status, body_text):
    """The trail row's code for one refusal: the gateway error's own code,
    else its type, else a plain http_<status>."""
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_obj(parsed):
        err = parsed.get("error")
        if is_obj(err):
            for key in ("code", "type"):
                if isinstance(err.get(key), str) and err[key] != "":
                    return err[key]
        if isinstance(parsed.get("code"), str) and parsed["code"] != "":
            return parsed["code"]
    if status == 0:
        return "network_error"
    return "http_%d" % status


def serve_with_failover(body, prompt_id, needs):
    """Dial the ordered candidates until one answers, or all refuse.

    Returns (model_id, upstream, trail): the model that answered and the
    refusals it took to get there, each {"status", "code", "model"} in the
    order the calls refused. A network failure counts as a refusal of that
    model with status 0 and moves on to the next candidate the same way.
    """
    trail = []
    for model in ordered_candidates(needs, prompt_id)[:MAX_DIALS_PER_PROMPT]:
        try:
            upstream = dial(model["id"], body)
        except Exception:
            trail.append({"status": 0, "code": "network_error", "model": model["id"]})
            note_refusal(model["id"])
            continue
        if upstream.ok:
            return model["id"], upstream, trail
        detail = upstream.text()
        trail.append(
            {"status": upstream.status, "code": refusal_code(upstream.status, detail), "model": model["id"]}
        )
        note_refusal(model["id"])
        if len(trail) >= MAX_TRAIL_ROWS:
            break
    return None, None, trail


# ------------------------------------------------------------------ the wires


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": LIST_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": LIST_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def openai_error(message, error_type, code):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def text_of(content):
    """The text of a string, or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if is_obj(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts)


def arguments_of(value):
    """A tool call's arguments as an object: the gateway's JSON string, or {}."""
    if is_obj(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_obj(parsed):
            return parsed
    return {}


def stop_reason_of(finish):
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body, model_id):
    """An Anthropic Messages request as a Chat Completions request.

    The system prompt becomes a system message, image blocks become image_url
    data URLs, tool_use blocks become tool_calls, tool_result blocks become
    tool messages, tools and tool_choice are rewritten, and the output cap,
    temperature, top_p, stop sequences and streaming pass through.
    """
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_obj(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            calls = []
            for block in content:
                if not is_obj(block) or block.get("type") != "tool_use":
                    continue
                calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_obj(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_obj(block):
                continue
            if block.get("type") == "tool_result":
                content_text = block.get("content")
                if isinstance(content_text, list):
                    content_text = text_of(content_text)
                    if content_text == "":
                        content_text = json.dumps(block["content"], separators=(",", ":"), ensure_ascii=False)
                elif not isinstance(content_text, str):
                    content_text = "" if content_text is None else json.dumps(
                        content_text, separators=(",", ":"), ensure_ascii=False
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": content_text,
                    }
                )
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image" and is_obj(block.get("source")):
                source_block = block["source"]
                if source_block.get("type") == "base64" and isinstance(source_block.get("data"), str):
                    media = source_block["media_type"] if isinstance(source_block.get("media_type"), str) else "image/png"
                    parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source_block["data"])}})
                elif source_block.get("type") == "url" and isinstance(source_block.get("url"), str):
                    parts.append({"type": "image_url", "image_url": {"url": source_block["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model_id, "messages": messages}
    if is_num(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    else:
        out["max_tokens"] = DEFAULT_MAX_TOKENS
    if is_num(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_num(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not is_obj(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                        "parameters": raw["input_schema"] if is_obj(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_obj(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body, model_id):
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_obj(choices[0]) else {}
    message = choice.get("message") if is_obj(choice.get("message")) else {}
    content = []
    text = message.get("content")
    if not isinstance(text, str):
        text = text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_obj(call) or not is_obj(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_obj(body.get("usage")) else {}
    return {
        "id": "msg_%s" % (body["id"] if isinstance(body.get("id"), str) else "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": as_int_or_zero(usage.get("prompt_tokens")),
            "output_tokens": as_int_or_zero(usage.get("completion_tokens")),
        },
    }


def sane_usage(answer):
    """A Chat Completions answer whose usage passes the wire checks.

    The gateway's usage passes through whole; only a missing or empty usage
    is repaired (input 0, output 1 when the answer carries text), because
    the wire requires an input count of at least 0 and an output of at least 1.
    """
    usage = answer.get("usage") if is_obj(answer.get("usage")) else {}
    prompt_tokens = as_int_or_zero(usage.get("prompt_tokens"))
    output_tokens = as_int_or_zero(usage.get("completion_tokens"))
    if prompt_tokens < 0:
        prompt_tokens = 0
    if output_tokens < 1:
        output_tokens = 1
    return {**answer, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": output_tokens}}


def sse_event(event, data):
    """One server-sent event: `event: <name>` when named, then the data line."""
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":"), ensure_ascii=False))


def split_frames(chunks):
    """SSE frames from upstream byte chunks, split on the blank line.

    A frame cut across reads is held until its end arrives; a body ending
    without one is yielded as it is. CRLF is normalized to LF.
    """
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            yield buffer[:at]
            buffer = buffer[at + 2 :]
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def rewrite_model_in_frame(frame, model_id):
    """One SSE frame with every data payload's `model` set to `model_id`.

    Comments, `event:` lines and non-JSON payloads pass through untouched;
    the [DONE] marker passes through untouched.
    """
    lines = []
    for line in frame.split("\n"):
        trimmed = line.strip()
        if not trimmed.startswith("data:"):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == "" or payload == "[DONE]":
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_obj(chunk):
            lines.append("data: %s" % json.dumps({**chunk, "model": model_id}, separators=(",", ":"), ensure_ascii=False))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n\n"


def openai_stream(chunks, model_id):
    """A Chat Completions SSE stream, frame by frame, model rewritten."""
    for frame in split_frames(chunks):
        yield rewrite_model_in_frame(frame, model_id)


def anthropic_stream(chunks, model_id):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per
    text block or tool call, then message_delta with the stop reason and
    usage, then message_stop. The stop reason is held until the stream's
    usage chunk has arrived, because OpenAI sends usage after finish_reason
    and Anthropic's message_delta carries the output tokens.
    """
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_slots = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": model_id,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    },
                },
            )
        ]

    def stop_open_blocks():
        nonlocal text_index
        out = []
        if text_index != -1:
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_slots.values()):
            out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        tool_slots.clear()
        return out

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *stop_open_blocks(),
            sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}},
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ]

    def events_of_frame(frame):
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out = []
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "[DONE]":
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else "end_turn"))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_obj(chunk):
                continue
            if is_obj(chunk.get("usage")):
                input_tokens = as_int_or_zero(chunk["usage"].get("prompt_tokens"))
                output_tokens = as_int_or_zero(chunk["usage"].get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            choice = choices[0] if choices and is_obj(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_obj(choice.get("delta")) else {}
            out.extend(start_events())
            if isinstance(delta.get("content"), str) and delta["content"] != "":
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event(
                            "content_block_start",
                            {"type": "content_block_start", "index": text_index, "content_block": {"type": "text", "text": ""}},
                        )
                    )
                out.append(
                    sse_event(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": text_index, "delta": {"type": "text_delta", "text": delta["content"]}},
                    )
                )
            calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
            for call in calls:
                if not is_obj(call):
                    continue
                upstream_index = call["index"] if is_num(call.get("index")) else len(tool_slots)
                index = tool_slots.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_slots[upstream_index] = index
                    fn = call.get("function") if is_obj(call.get("function")) else {}
                    out.append(
                        sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else "toolu_%d" % index,
                                    "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                                    "input": {},
                                },
                            },
                        )
                    )
                fn = call.get("function") if is_obj(call.get("function")) else {}
                if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                    out.append(
                        sse_event(
                            "content_block_delta",
                            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                        )
                    )
            if choice.get("finish_reason") is not None:
                pending_stop = stop_reason_of(choice.get("finish_reason"))
        return out

    for frame in split_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else "end_turn"):
            yield event


# ------------------------------------------------------------------ the server


def trail_header_value(trail):
    """The failures trail as one header value, or None when there is none."""
    if not trail:
        return None
    rows = trail[:MAX_TRAIL_ROWS]
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    if len(text.encode("utf-8")) > 8192:
        return None
    return text


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "vocaine-aggregator02/1"
    sys_version = ""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    # -- plumbing -----------------------------------------------------

    def dispatch(self):
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error(wire, 404, "not_found", "no route %s %s" % (self.command, path))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status, body, trail=None):
        text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        header = trail_header_value(trail) if trail else None
        if header is not None:
            self.send_header(FAILURES_HEADER, header)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error(self, wire, status, error_type, message, trail=None):
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message), trail)
        else:
            self.send_json(status, openai_error(message, error_type, None), trail)

    def begin_stream(self, trail=None):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        # http.server does not chunk a response without a length; the close
        # is what ends the stream for the client.
        self.send_header("connection", "close")
        header = trail_header_value(trail) if trail else None
        if header is not None:
            self.send_header(FAILURES_HEADER, header)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self):
        """One client request body, or None when it is not one JSON object."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_REQUEST_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_obj(value) else None

    def refused(self, wire, trail):
        """Every candidate refused: a 502 in the client's wire, with the trail."""
        last = trail[-1] if trail else {"status": 502, "code": "upstream_error", "model": None}
        detail = "the gateway answered HTTP %s (%s)" % (last["status"], last["code"])
        if wire == "anthropic":
            self.send_error(wire, 502, "api_error", detail, trail)
        else:
            self.send_error(wire, 502, "upstream_error", detail, trail)

    # -- the routes ---------------------------------------------------

    def models(self):
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = catalog_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self):
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by
        the best live free model for this prompt, with failover."""
        body = self.read_json()
        if body is None:
            self.send_error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        prompt_id, needs = prompt_row_for(body)
        gateway_body = {**body, "model": "auto"}
        if not is_num(gateway_body.get("max_tokens")) and not is_num(gateway_body.get("max_completion_tokens")):
            gateway_body["max_tokens"] = DEFAULT_MAX_TOKENS
        try:
            model_id, upstream, trail = serve_with_failover(gateway_body, prompt_id, needs)
        except Exception as err:
            self.send_error("openai", 502, "upstream_unreachable", str(err))
            return
        if upstream is None:
            self.refused("openai", trail)
            return
        if body.get("stream") is True:
            self.begin_stream(trail)
            try:
                for frame in openai_stream(upstream.chunks(), model_id):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_obj(parsed):
            self.send_error("openai", 502, "upstream_error", "the gateway answered a body that is not JSON", trail)
            return
        self.send_json(200, sane_usage({**parsed, "model": model_id}), trail)

    def messages(self):
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat
        Completions and answered back in the Messages wire."""
        body = self.read_json()
        if body is None:
            self.send_error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        prompt_id, needs = prompt_row_for(body)
        gateway_body = anthropic_to_openai(body, "auto")
        try:
            model_id, upstream, trail = serve_with_failover(gateway_body, prompt_id, needs)
        except Exception as err:
            self.send_error("anthropic", 502, "api_error", str(err))
            return
        if upstream is None:
            self.refused("anthropic", trail)
            return
        if body.get("stream") is True:
            self.begin_stream(trail)
            try:
                for event in anthropic_stream(upstream.chunks(), model_id):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_obj(parsed):
            self.send_error("anthropic", 502, "api_error", "the gateway answered a body that is not JSON", trail)
            return
        self.send_json(200, openai_to_anthropic(sane_usage(parsed), model_id), trail)


# ------------------------------------------------------------------ register + poll


def register(port):
    """POST /register with the port and the one catalog; the env answers
    with the trial's prompt count."""
    payload = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        "%s/register" % ENV_URL,
        data=payload.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(
            "POST %s/register answered HTTP %d: %s" % (ENV_URL, err.code, err.read().decode("utf-8", "replace")[:500])
        ) from err
    if not 200 <= status < 300:
        raise RuntimeError("POST %s/register answered HTTP %d: %s" % (ENV_URL, status, reply_text[:500]))
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_obj(reply) and is_num(reply.get("requests")):
        log("registered: the trial holds %s prompts" % reply["requests"])


def poll_work():
    """Poll GET /work until it says done, then exit 0 at once."""
    while True:
        time.sleep(POLL_EVERY_SEC)
        try:
            with urllib.request.urlopen("%s/work" % ENV_URL, timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_obj(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_args):
    """SIGTERM and SIGINT end the trial: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


# ------------------------------------------------------------------ start


def start():
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print
        # the one built catalog text and stop before any server starts.
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if gateway_chat_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log("listening on 0.0.0.0:%d with %d free models" % (port, len(catalog_models())))
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll_work, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main():
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
