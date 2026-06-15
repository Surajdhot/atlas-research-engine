# CLAUDE.md

## Project identity
Atlas is a multi-agent research engine. A planner agent decomposes a research
question into sub-questions, specialist retrieval agents gather evidence from
web search, Wikipedia, and arXiv, and a synthesis agent combines the evidence
into a cited report with per-claim confidence scores. It flags contradictions
between sources rather than hiding them.

## Code style rules
- Python only.
- All functions must have docstrings.
- Type hints on every function signature.
- Max function length: 40 lines. Split if longer.
- No print statements — use the logging module everywhere.
- Error handling: never use bare except. Always catch specific exceptions.
- Constants go in config.py, never hardcoded in logic files.
- Use async/await throughout — all agent and API calls are async.

## Architecture decisions — do not change these
- Each agent is its own class in the agents/ directory.
- All LLM calls live in llm_client.py only.
- Each external data source has its own tool module in tools/
  (web_search.py, wikipedia.py, arxiv.py).
- All prompts live in prompts/ as .txt files — never hardcoded in Python.
- Agent orchestration logic lives in orchestrator.py only.
- Configuration comes from .env via python-dotenv — no hardcoded secrets.

## What NOT to do
- Do not use LangChain or CrewAI — build the multi-agent system from scratch
  using a raw OpenAI-compatible LLM SDK with tool use. This is deliberate: I want
  to demonstrate I understand agent orchestration without a framework.
- Do not create a heavy frontend — a simple Streamlit UI is the only UI.
- Do not write placeholder, stub, or TODO code — everything must work.
- Do not commit .env — only .env.example.

## Testing
- Every module needs a corresponding test file in tests/.
- Use pytest with pytest-asyncio.
- Mock all external API calls in tests — no real API calls in the test suite.

## Git commit style
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:
- One logical change per commit — never commit everything at once.
- Commit after each component works.
