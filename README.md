# vocaine-aggregator02

One server, two client wires, fourteen live free models — an entry for
make.fun's `free-model-aggregator-02` ($500).

The server speaks OpenAI Chat Completions (`POST /v1/chat/completions`) and
Anthropic Messages (`POST /v1/messages`), in JSON and in server-sent events,
and answers every prompt with a real completion from one of the fourteen
free OpenRouter models in the trial brief, dialled through the grader's
gateway. `GET /v1/models` answers in the wire its auth header names
(`Authorization: Bearer` → OpenAI shape, `x-api-key` → Anthropic shape).

## Run

```sh
python3 main.py --list   # print the catalog document, byte for byte as registered
python3 main.py          # serve: register with $MAKEFUN_ENV_URL, exit 0 when /work says done
```

Python 3.13 standard library only; nothing is installed before it runs.

## Routing (vocaine's strategy)

Per-prompt specialization plus deterministic weighted round-robin:

- every candidate model carries a live score, `rating_milli × reliability`,
  where reliability starts at 1 and halves on each refusal that model
  suffers during the trial;
- a streaming prompt takes the highest live score over all free models;
- a JSON-mode prompt takes the highest live score over JSON-capable models
  only; a tools prompt over tools-capable models only;
- when several models tie on the top score, their order is a round-robin
  rotation whose offset is `sha256(trial_seed + "|" + prompt_id)` — no wall
  clock, no unseeded randomness, so a replay is byte-identical;
- on a refusal the server fails over to the next model in that same rotated
  order (at most 6 dials a prompt) and reports the refusals it recovered
  from in the `x-makefun-upstream-failures` header.

The `model` field on every answer and every streamed chunk names the model
really dialled.
