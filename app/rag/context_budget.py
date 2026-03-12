"""
context_budget.py — Token-aware context trimming for Ollama.

CPU-only deployment: num_ctx=2048.
KV cache scales linearly with num_ctx. On CPU:
  llama3.2 @ 4096 ctx: ~0.45 GB KV cache + 1.87 GB weights = 2.32 GB
  llama3.2 @ 2048 ctx: ~0.22 GB KV cache + 1.87 GB weights = 2.09 GB  ← saves 230MB
Two Ollama lanes (ask + pipeline) at 2048 each = 4.18 GB total — safe on 24 GB.

Budget allocation (out of 2048 total context):
  Fixed overhead (MEASURED: system_prompt=781t + user_template=153t + margin): ~950 tokens
  Remaining for content: 1098 tokens, allocated as:
    Focused track context : 350 (most relevant — always include first)
    Video summaries       : 100
    Memory graph          : 100
    Behaviour analysis    :  80
    Timeline              : 120
    Raw events            :  80
    Scene captions        : 200 (moondream descriptions + batch captions)
    Safety margin         :  68

Rough token estimate: 1 token ~= 3.5 chars (conservative for English surveillance text)
"""

from app.core.logging import get_logger

_LOG = get_logger()

_CHARS_PER_TOKEN = 3.5

# Context window to request from Ollama.
# KV cache cost = num_ctx × 2 × n_layers × hidden_dim × 2 bytes
# llama3.2: 28 layers, 2048 dim → 4096 ctx costs ~0.94 GB vs 8192 → ~1.88 GB
OLLAMA_NUM_CTX = 2048

# Budget per section in tokens (must sum to < OLLAMA_NUM_CTX - overhead)
_BUDGETS = {
    "focused":    350,   # primary: temporal track context
    "summary":    100,   # per-window narrative summaries
    "memory":     100,   # knowledge graph nodes
    "behaviour":   80,   # temporal behaviour labels
    "timeline":   120,   # per-second event spine
    "raw_events":  80,   # raw entry/exit/dwell rows
    "captions":   200,   # scene captions (batch) + moondream descriptions (live)
    # sum=1030, + 950 overhead + 68 margin = 2048 ✓
}

# Fixed overhead estimate (system prompt + template text + question)
_FIXED_OVERHEAD_TOKENS = 950  # measured: system_prompt(781t) + user_template(153t) + margin


def _token_estimate(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def trim_to_budget(text: str, max_tokens: int, label: str = "") -> str:
    """
    Trim text to max_tokens by cutting from the middle (keeps head and tail).
    Cuts are done at newline boundaries to avoid breaking sentences.
    """
    if not text:
        return ""
    estimated_tokens = _token_estimate(text)
    if estimated_tokens <= max_tokens:
        return text

    _LOG.info("qa_context_trimmed", section=label or "unknown", estimated_tokens=estimated_tokens, max_tokens=max_tokens)

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
    summary: str = "",
    memory: str = "",
    behaviour: str = "",
    timeline: str = "",
    raw_events: str = "",
    captions: str = "",
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
    if summary:
        trimmed = trim_to_budget(summary, _BUDGETS["summary"], "video_summary")
        sections.append(trimmed)
    if captions:
        trimmed = trim_to_budget(captions, _BUDGETS["captions"], "scene_captions")
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
    final_tokens = _token_estimate(result)
    if final_tokens > total_budget:
        _LOG.info("qa_context_total_trimmed", estimated_tokens=final_tokens, max_tokens=total_budget)
        result = trim_to_budget(result, total_budget, "TOTAL_CONTEXT")

    return result