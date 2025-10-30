import logging
from typing import Any


def ensure_root_logger(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)
    root.setLevel(level)


def fmt_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts: list[str] = []
    for k, v in fields.items():
        try:
            parts.append(f"{k}={repr(v)}")
        except Exception:
            parts.append(f"{k}=<unrepr>")
    return " ".join(parts)


def log_start(node: str, **fields: Any) -> None:
    logging.info("")
    logging.info("[%s] START %s", node, fmt_fields(fields))


def log_end(node: str, **fields: Any) -> None:
    logging.info("[%s] END %s", node, fmt_fields(fields))
    logging.info("")


