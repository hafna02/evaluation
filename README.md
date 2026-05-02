# Greenwashing Detector 🌱

A **multi-agent compound AI system** that detects greenwashing in corporate text and decides — for every flagged phrase — whether to **keep**, **replace**, or **delete** it. Runs entirely on **small offline models** (Ollama / HuggingFace) and is **fully traced and evaluated with LangSmith**.

Built as an end-to-end reference for:

- 🔎 **LangSmith tracing** of a multi-agent pipeline
- 🧪 **Three flavours of evaluators**: custom rule-based, LangChain built-in, and **LLM-as-judge**
- 🛡️ **Failure handling**: retries, JSON repair, validation gates, conservative fallbacks
- 🤖 **Agent orchestration** with LangGraph (supervisor + conditional routing)
- 💻 **100% local** — no OpenAI key required

---

## Architecture

```
                       ┌──────────────┐
                       │   Document   │
                       └──────┬───────┘
                              ▼
                  ┌────────────────────────┐
                  │   Detector Agent       │  finds suspicious phrases
                  │   → DetectorOutput     │
                  └────────────┬───────────┘
                               │  (no spans? → END)
                               ▼
                  ┌────────────────────────┐
                  │   Classifier Agent     │  keep / replace / delete
                  │   → ClassifierOutput   │
                  └────────────┬───────────┘
                               ▼
                  ┌────────────────────────┐
                  │   Rewriter Agent       │  honest replacements
                  │   → RewriterOutput     │  + final document
                  └────────────┬───────────┘
                               ▼
                       ┌──────────────┐
                       │ AnalysisResult│
                       └──────────────┘
```

Each agent is a LangGraph node. Routing is conditional on detector output — if nothing is flagged, the graph short-circuits and skips the downstream agents. Every node is decorated with `@traceable`, so the full call tree appears in LangSmith with per-node latency, tokens, and inputs/outputs.

---
## Project layout

```
greenwashing-detector/
├── src/greenwashing/
│   ├── agents/
│   │   ├── detector.py      # finds greenwashing phrases
│   │   ├── classifier.py    # decides keep/replace/delete
│   │   └── rewriter.py      # generates honest replacements
│   ├── graph.py             # LangGraph wiring + analyse() entrypoint
│   ├── models.py            # Ollama default + HF fallback
│   ├── schemas.py           # Pydantic schemas (typed agent IO)
│   ├── failure.py           # safe_invoke: retries + JSON repair
│   ├── tracing.py           # LangSmith bootstrap
│   ├── config.py            # pydantic-settings
│   └── cli.py               # `greenwash` Typer CLI
├── evals/
│   ├── dataset.py           # seed LangSmith with synthetic examples
│   ├── custom_evaluators.py # rule-based: precision, recall, accuracy, ...
│   ├── llm_judge.py         # LLM-as-judge: residue + faithfulness
│   └── run_eval.py          # the full eval runner
├── data/seed_examples.json  # 20 labelled examples
├── notebooks/01_walkthrough.ipynb
└── pyproject.toml
```

---

## Evaluation

Seed the dataset and run the full suite:

```bash
python -m evals.run_eval --seed
```

This evaluates the pipeline against `data/seed_examples.json` with:

| Evaluator | Type | What it measures |
|---|---|---|
| `detection_recall` | Custom | % of expected greenwashing phrases the detector caught |
| `detection_precision` | Custom | % of detected phrases that were actually expected |
| `action_accuracy` | Custom | Did classifier pick correct action (keep/replace/delete)? |
| `output_validity` | Custom | Did pipeline return a non-empty document with no errors? |
| `removed_flagged_terms` | Custom | Did the final text actually drop the flagged phrases? |
| `embedding_distance` | LangChain built-in | Semantic distance between original and rewrite |
| `judge_greenwashing_residue` | LLM-as-judge | 1-5 Likert: how clean is the rewrite? |
| `judge_faithfulness` | LLM-as-judge | Did the rewrite invent claims that weren't in the original? |

Results stream to your LangSmith project — open the experiments tab to compare runs, drill into individual examples, and inspect failure modes.

---

## Failure handling

The pipeline is built to **never crash on bad LLM output**:

- `safe_invoke` retries `MAX_RETRIES` times with JSON extraction + Pydantic validation
- Malformed responses are repaired (code-fence stripping, brace matching)
- On total failure, each agent has a conservative fallback (e.g. classifier defaults to `keep`)
- All errors are appended to `state["errors"]` and surface in the final `AnalysisResult` and LangSmith trace

Try it: pass an empty string or garbage to `analyse()` and you still get a typed result back.

---

## Configuration

All settings come from `.env` via `pydantic-settings`. Highlights:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `huggingface` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Agent model |
| `OLLAMA_JUDGE_MODEL` | `llama3.2:3b` | Judge model — bump to `llama3.1:8b` for stricter judging |
| `MAX_RETRIES` | `2` | Per-agent retry budget |
| `LANGSMITH_TRACING` | `true` | Set `false` to run fully offline |

---

## Why this project?

LangSmith documentation tends to use OpenAI + a single chain. This repo demonstrates a more realistic setup: a **multi-agent graph**, **fully local models**, and **all three evaluator categories** working together on a use case that actually matters (regulators are increasingly enforcing against vague green claims — see the EU's Green Claims Directive).

---

## License

MIT
