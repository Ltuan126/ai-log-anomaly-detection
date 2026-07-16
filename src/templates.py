"""
Match raw HDFS log lines against the 29 known event templates
(data/raw/preprocessed/HDFS.log_templates.csv) so production inference can
turn free-text log content into an EventId -- the same feature space
anomaly_model_full.pkl was trained on (Event_occurrence_matrix.csv).

This is template *matching*, not full Drain parsing: HDFS's log format is a
small, fixed set of templates (this is the same 29-template set the
original loghub/loglizer authors used), so matching against known templates
is equivalent to running the parser, without needing to depend on a
separate Drain implementation in production.
"""

import csv
import re
from pathlib import Path


def _compile_template(template: str) -> re.Pattern:
    """Turn a "[*]literal[*]" template into a regex, wildcards -> .*?"""
    parts = template.split("[*]")
    pattern = ".*?".join(re.escape(part) for part in parts)
    return re.compile(pattern, re.IGNORECASE)


class TemplateMatcher:
    def __init__(self, templates_path: Path):
        if not templates_path.exists():
            raise FileNotFoundError(f"Template file not found at {templates_path}")

        rows = []
        with templates_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["EventId"], row["EventTemplate"]))

        # Try the most specific (longest literal content) templates first so
        # a generic template can't shadow a more precise one.
        compiled = [
            (len(tmpl.replace("[*]", "")), event_id, _compile_template(tmpl))
            for event_id, tmpl in rows
        ]
        compiled.sort(key=lambda item: -item[0])
        self._compiled = compiled
        self.event_ids = [event_id for event_id, _ in rows]

    def match(self, content: str) -> str | None:
        for _, event_id, regex in self._compiled:
            if regex.search(content):
                return event_id
        return None
