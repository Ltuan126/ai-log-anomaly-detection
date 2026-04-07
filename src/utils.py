import json
import logging
import os
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
	"""Render logs in JSON for machine-friendly ingestion."""

	def format(self, record: logging.LogRecord) -> str:
		payload = {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"level": record.levelname,
			"logger": record.name,
			"message": record.getMessage(),
		}

		for key, value in record.__dict__.items():
			if key.startswith("_") or key in {
				"name",
				"msg",
				"args",
				"levelname",
				"levelno",
				"pathname",
				"filename",
				"module",
				"exc_info",
				"exc_text",
				"stack_info",
				"lineno",
				"funcName",
				"created",
				"msecs",
				"relativeCreated",
				"thread",
				"threadName",
				"processName",
				"process",
			}:
				continue
			payload[key] = value

		if record.exc_info:
			payload["exception"] = self.formatException(record.exc_info)

		return json.dumps(payload, ensure_ascii=True)


def configure_logging(default_level: str = "INFO") -> None:
	level_name = os.getenv("LOG_LEVEL", default_level).upper()
	level = getattr(logging, level_name, logging.INFO)

	root_logger = logging.getLogger()
	root_logger.setLevel(level)

	handler = logging.StreamHandler()
	handler.setFormatter(JsonFormatter())

	root_logger.handlers = [handler]
