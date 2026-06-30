"""Logging estruturado do projeto.

Usado em todo o código no lugar de `print()`, conforme exigido pelas boas
práticas do desafio. Emite logs em formato JSON, facilitando ingestão por
ferramentas de observabilidade (ELK, CloudWatch, etc.) em produção.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formata cada registro de log como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Campos extras (ex: logger.info("msg", extra={"latency_ms": 12}))
        reserved = logging.LogRecord(
            "", 0, "", 0, "", None, None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retorna um logger configurado com saída JSON em stdout.

    Idempotente: chamadas repetidas com o mesmo `name` não duplicam handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
