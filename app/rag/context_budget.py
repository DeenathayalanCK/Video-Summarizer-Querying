"""
context_budget.py — Token-aware context trimming for Ollama.

Ollama's default num_ctx is 4096 for most models (llama3, mistral etc).
We pass options.num_ctx=8192 in every request to use the model's full window.
Even so, we must stay within budget — raw events alone can exceed 8k.

Budget allocation (out of 8192 total context):
  Fixed overhead (system prompt + template + question): ~800 tokens
  Remaining for content: 7392 tokens, allocated as:
    Focused track context : 2000 (most relevant — always include first)
    Memory graph          : 1000
    Behaviour analysis    :  700
    Timeline              :  800
    Raw events            : remainder (capped at 2500)

Rough token estimate: 1 token ~= 3.5 chars (conservative for English surveillance text)
"""

_CHARS_PER_TOKEN = 3.5

# Context window to request from Ollama
OLLAMA_NUM_CTX = 8192

# Budget per section in tokens
_BUDGETS = {
    "focused":    2000,
    "memory":     1000,
    "behaviour":   700,
    "timeline":    800,
    "raw_events": 2500,
}

# Fixed overhead estimate (system prompt + template text + question)
_FIXED_OVERHEAD_TOKENS = 900


def _token_estimate(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def trim_to_budget(text: str, max_tokens: int, label: str = "") -> str:
    """
    Trim text to max_tokens by cutting from the middle (keeps head and tail).
    Cuts are done at newline boundaries to avoid breaking sentences.
    """
    if not text:
        return ""
    if _token_estimate(text) <= max_tokens:
        return text

    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    head_chars = int(max_chars * 0.6)
    tail_chars = int(max_chars * 0.3)

    lines = text.split("\n")
    result = []
    chars = 0
    # Take from head
    for line in lines:
        if chars + len(line) > head_chars:
            break
        result.append(line)
        chars += len(line) + 1

    cut_notice = f"\n  ... [TRIMMED: {label} exceeded budget of {max_tokens} tokens] ...\n"
    result.append(cut_notice)

    # Take from tail
    tail_lines = []
    tail_chars_used = 0
    for line in reversed(lines):
        if tail_chars_used + len(line) > tail_chars:
            break
        tail_lines.insert(0, line)
        tail_chars_used += len(line) + 1
    result.extend(tail_lines)

    return "\n".join(result)


def build_budgeted_context(
    focused: str = "",
    memory: str = "",
    behaviour: str = "",
    timeline: str = "",
    raw_events: str = "",
) -> str:
    """
    Assemble all context sections respecting per-section token budgets.
    Returns the final context string guaranteed to stay within 
    (OLLAMA_NUM_CTX - _FIXED_OVERHEAD_TOKENS) tokens.
    """
    sections = []

    if focused:
        trimmed = trim_to_budget(focused, _BUDGETS["focused"], "focused_track_context")
        sections.append(trimmed)

    if memory:
        trimmed = trim_to_budget(memory, _BUDGETS["memory"], "memory_graph")
        sections.append(trimmed)

    if behaviour:
        trimmed = trim_to_budget(behaviour, _BUDGETS["behaviour"], "behaviour_analysis")
        sections.append(trimmed)

    if timeline:
        trimmed = trim_to_budget(timeline, _BUDGETS["timeline"], "video_timeline")
        sections.append(trimmed)

    if raw_events:
        trimmed = trim_to_budget(raw_events, _BUDGETS["raw_events"], "raw_events")
        sections.append(trimmed)

    result = "\n\n".join(s for s in sections if s)

    # Final safety net — if we somehow still exceed, hard-cut
    total_budget = OLLAMA_NUM_CTX - _FIXED_OVERHEAD_TOKENS
    if _token_estimate(result) > total_budget:
        result = trim_to_budget(result, total_budget, "TOTAL_CONTEXT")

    return result