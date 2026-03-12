"""
ollama_logger.py — Thin rotating logger for every Ollama API call.

Writes one JSON line per call to /data/logs/ollama_calls.jsonl.
Fields per entry:
  ts          — ISO timestamp
  call_type   — ask | summary | attribute | embed | caption | activity
  model       — model name used
  prompt      — first 800 chars of prompt sent
  response    — first 500 chars of response received
  elapsed_ms  — wall-clock milliseconds for the call
  status      — ok | timeout | error
  error       — error message if status != ok

Log file rotates at 10 MB, keeps 3 backups.
View via GET /api/v1/ollama-log?n=100
"""

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Optional


_LOG_DIR  = Path(os.getenv("OLLAMA_LOG_DIR", "/data/logs"))
_LOG_FILE = _LOG_DIR / "ollama_calls.jsonl"
_MAX_BYTES  = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT = 3

_PROMPT_LIMIT   = 800   # chars stored for prompt
_RESPONSE_LIMIT = 500   # chars stored for response

# One-time setup — safe to call multiple times (handler checked by name)
_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger("ollama_calls")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False   # don't bubble up to root logger

    if not lg.handlers:
        handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)

    _logger = lg
    return lg


def log_call(
    *,
    call_type: str,          # ask | summary | attribute | embed | caption | activity
    model: str,
    prompt: str,
    response: str = "",
    elapsed_ms: float = 0.0,
    status: str = "ok",      # ok | timeout | error
    error: str = "",
) -> None:
    """Write one JSONL entry for an Ollama call. Never raises."""
    try:
        entry = {
            "ts":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "call_type":   call_type,
            "model":       model,
            "prompt":      prompt[:_PROMPT_LIMIT],
            "response":    response[:_RESPONSE_LIMIT],
            "elapsed_ms":  round(elapsed_ms, 1),
            "status":      status,
        }
        if error:
            entry["error"] = error[:300]
        _get_logger().info(json.dumps(entry, ensure_ascii=False))
    except Exception:
        pass   # logging must never crash the pipeline


class OllamaCallTimer:
    """
    Context manager that times a block and logs the result.

    Usage:
        with OllamaCallTimer(call_type="ask", model="llama3.2", prompt=prompt) as t:
            response = requests.post(...)
            t.response = response.json().get("response", "")
        # log entry written automatically on __exit__
    """

    def __init__(self, *, call_type: str, model: str, prompt: str):
        self.call_type = call_type
        self.model     = model
        self.prompt    = prompt
        self.response  = ""
        self.status    = "ok"
        self.error     = ""
        self._t0: float = 0.0

    def __enter__(self) -> "OllamaCallTimer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.monotonic() - self._t0) * 1000
        if exc_type is not None:
            import requests as _req
            if exc_type.__name__ == "Timeout" or (
                hasattr(_req.exceptions, "Timeout") and
                issubclass(exc_type, _req.exceptions.Timeout)
            ):
                self.status = "timeout"
            else:
                self.status = "error"
            self.error = str(exc_val)[:300]
        log_call(
            call_type=self.call_type,
            model=self.model,
            prompt=self.prompt,
            response=self.response,
            elapsed_ms=elapsed_ms,
            status=self.status,
            error=self.error,
        )
        return False   # don't suppress exceptions


def read_recent_entries(n: int = 100) -> list[dict]:
    """
    Read the last N entries from the log file.
    Returns a list of dicts, newest-first.
    Called by GET /api/v1/ollama-log.
    """
    try:
        if not _LOG_FILE.exists():
            return []
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
        # Take last N non-empty lines, reverse so newest first
        recent = [l for l in lines if l.strip()][-n:]
        result = []
        for line in reversed(recent):
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result
    except Exception:
        return []