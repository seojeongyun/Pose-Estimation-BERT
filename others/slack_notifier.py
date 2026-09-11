import json
import os
import queue
import re
import sys
import threading
from urllib.request import Request, urlopen

class SlackEpochNotifier:
    """Post selected, path-sanitized training lines once per epoch."""

    WEBHOOK_PREFIX = "https://hooks.slack.com/services/"
    ABSOLUTE_PATH_PATTERN = re.compile(r"(?<!\w)/(?:[^\s|]+/)+[^\s|]*")

    def __init__(self, run_name, header=None):
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
        self.webhook_url = (
            webhook_url if webhook_url.startswith(self.WEBHOOK_PREFIX) else ""
        )
        self.run_name = str(run_name)
        self.header = str(header) if header else self.run_name
        self.enabled = bool(self.webhook_url)

        self._partial = ""
        self._epoch_lines = []
        self._collecting = False
        self._queue = queue.Queue()
        self._worker = None

        if self.enabled:
            self._worker = threading.Thread(
                target=self._send_worker,
                name="slack-epoch-notifier",
                daemon=True,
            )
            self._worker.start()

    def write(self, text):
        if not self.enabled:
            return

        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._process_line(line)

    def _process_line(self, line):
        if "TRAIN START" in line:
            if self._epoch_lines:
                self._flush_epoch()
            self._collecting = True
            self._epoch_lines.append(line)
            return

        if not self._collecting:
            return

        keep_line = (
            line.startswith("[TRAIN][Step")
            or line.startswith("[TRAIN][Epoch")
            or line.startswith("Throughput:")
            or line.startswith("Elapsed time per epoch:")
            or "VALID START" in line
            or line.startswith("[VALID]")
            or line.startswith("Expected:")
            or line.startswith("Predicted:")
            or line.startswith("Final Epoch Model save")
        )
        if not keep_line:
            return

        line = self._sanitize_line(line)
        self._epoch_lines.append(line)
        if line.startswith("Final Epoch Model save"):
            self._flush_epoch()

    def _sanitize_line(self, line):
        if "Condition PR threshold analysis:" in line:
            line = line.split(" | exercise thresholds:", 1)[0]
        if ": /" in line:
            line = line.split(": /", 1)[0] + ": [saved locally]"
        return self.ABSOLUTE_PATH_PATTERN.sub("[path removed]", line)

    def _flush_epoch(self):
        if not self._epoch_lines:
            return

        body = "\n".join(self._epoch_lines)
        message = "*{}*\n_Run: {}_\n```{}```".format(
            self.header,
            self.run_name,
            body,
        )
        self._queue.put(message)
        self._epoch_lines = []
        self._collecting = False

    def _send_worker(self):
        while True:
            message = self._queue.get()
            if message is None:
                return

            try:
                payload = json.dumps(
                    {"text": message},
                    ensure_ascii=False,
                ).encode("utf-8")
                request = Request(
                    self.webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    response.read()
            except Exception as error:
                error_stream = getattr(sys, "__stderr__", None)
                if error_stream is not None:
                    error_stream.write(
                        "[Slack notification failed] {}\n".format(error))
                    error_stream.flush()

    def close(self):
        if not self.enabled:
            return

        if self._partial:
            self._process_line(self._partial)
            self._partial = ""
        self._flush_epoch()
        self._queue.put(None)

        if self._worker is not None:
            self._worker.join(timeout=3)
