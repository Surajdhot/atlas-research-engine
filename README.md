# 🧭 Atlas — Multi-Agent Research Engine

Atlas answers any research question by coordinating a team of specialist agents.
A **planner** decomposes the question into focused sub-questions, **retrieval
agents** gather evidence from the web, Wikipedia, and arXiv *in parallel*, and a
**synthesis agent** combines the findings into a cited report with a confidence
score for every claim — explicitly flagging contradictions between sources
rather than hiding them.

It is built from scratch on the raw **Anthropic SDK** with tool use — no
LangChain, no CrewAI — to demonstrate multi-agent orchestration without a
framework.

## How it works

```
                ┌─────────────┐
   question ───▶│   Planner   │  decomposes into 2–5 sub-questions
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐   asyncio.gather  → runs concurrently
        ▼              ▼              ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │Retrieval │   │Retrieval │   │Retrieval │   each agent decides which
  │  agent   │   │  agent   │   │  agent   │   sources to use via Claude
  └────┬─────┘   └────┬─────┘   └────┬─────┘   tool-use (web/wiki/arxiv)
       └──────────────┼──────────────┘
                      ▼
                ┌─────────────┐
                │  Synthesis  │  claims + confidence + conflict flags
                └──────┬──────┘
                       ▼
                  cited Report
```

The **parallel retrieval** stage is the technical centrepiece: one retrieval
agent runs per sub-question, and they execute concurrently via
`asyncio.gather` rather than one after another.

## Project structure

```
atlas-research-engine/
├── CLAUDE.md              # project conventions for Claude Code
├── app.py                 # Streamlit UI
├── config.py              # constants, settings, env validation
├── llm_client.py          # all Claude API calls (tool-use loop + structured output)
├── orchestrator.py        # coordinates the agents (parallel retrieval)
├── models.py              # dataclasses: SubQuestion, Evidence, Claim, Report
├── agents/
│   ├── base_agent.py      # shared base class
│   ├── planner_agent.py   # question → sub-questions
│   ├── retrieval_agent.py # sub-question → evidence (dynamic tool choice)
│   └── synthesis_agent.py # evidence → claims + report
├── tools/
│   ├── web_search.py      # Tavily API
│   ├── wikipedia.py       # Wikipedia REST API
│   └── arxiv.py           # arXiv Atom API
├── prompts/               # planner.txt, retrieval.txt, synthesis.txt
└── tests/                 # pytest suite (all external calls mocked)
```

## Setup

Requires Python 3.11+ (3.9+ works for local development).

```bash
git clone https://github.com/Surajdhot/atlas-research-engine.git
cd atlas-research-engine

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then fill in your keys
```

### Environment variables

| Variable            | Required | Description                                   |
| ------------------- | -------- | --------------------------------------------- |
| `ANTHROPIC_API_KEY` | yes      | Anthropic API key for the agents.             |
| `TAVILY_API_KEY`    | yes      | [Tavily](https://tavily.com) key for web search. |
| `MODEL`             | no       | Claude model id (default `claude-fable-5`).   |
| `LOG_LEVEL`         | no       | Logging level (default `INFO`).               |

Wikipedia and arXiv need no API key.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, enter a research question, and click **Research**.
The UI streams progress through planning, retrieval, and synthesis, then shows
each claim with a colour-coded confidence bar (green > 0.7, amber 0.4–0.7,
red < 0.4), its supporting and conflicting sources, and an overall confidence
score.

### Docker

```bash
docker compose up --build
```

This builds the image and serves the app on http://localhost:8501, reading
secrets from your `.env` file.

## Testing

```bash
pytest tests/
```

All external API calls (Claude, Tavily, Wikipedia, arXiv) are mocked — the suite
makes no network requests. It covers planner output bounds, synthesis conflict
handling and confidence averaging, and that retrieval genuinely runs in parallel.

## Design notes

- **No framework.** The multi-agent system is built directly on the Anthropic
  SDK with tool use. Each agent is its own class under `agents/`.
- **All Claude calls are isolated** in `llm_client.py`; every other module stays
  provider-agnostic.
- **Prompts live in `prompts/`** as plain text, never hardcoded in Python.
- **Async throughout** — every agent and API call is `async`, enabling the
  concurrent retrieval fan-out.
- **Confidence & conflicts.** Each claim must cite supporting evidence (claims
  with none are dropped) and conflicting evidence lowers a claim's confidence,
  so disagreements between sources surface instead of being hidden.
