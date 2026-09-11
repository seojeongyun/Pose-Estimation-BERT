import atexit
import json
import os
import platform
import shlex
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from others.slack_notifier import SlackEpochNotifier


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _jsonable(item) for key, item in vars(value).items()}
    return str(value)


def _write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(str(temporary_path), str(path))


def _allocate_run_directory(logging_root, started_at, run_name=None):
    logging_root = Path(logging_root)
    logging_root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_name = str(run_name)
        if Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError(
                "run_name must be a single directory name: {}".format(
                    run_name
                )
            )
        base_name = "{}_{}".format(
            started_at.strftime("%m%d_%H%M"),
            run_name,
        )
    else:
        # run_name을 지정하지 않은 기존 호출은 종전 형식을 유지한다.
        base_name = started_at.strftime("%Y%m%d_%H%M%S")

    candidate = logging_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = logging_root / "{}_{:02d}".format(base_name, suffix)
        suffix += 1

    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


class _TeeStream:
    """stdout/stderr를 터미널과 train.log에 동시에 기록한다."""

    def __init__(self, terminal_stream, log_handle, lock,
                 slack_notifier=None):
        self.terminal_stream = terminal_stream
        self.log_handle = log_handle
        self.lock = lock
        self.slack_notifier = slack_notifier

    @property
    def encoding(self):
        return getattr(self.terminal_stream, "encoding", "utf-8")

    def write(self, text):
        with self.lock:
            self.terminal_stream.write(text)
            self.log_handle.write(text)
            self.log_handle.flush()
            if self.slack_notifier is not None:
                self.slack_notifier.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal_stream.flush()
            self.log_handle.flush()

    def isatty(self):
        return self.terminal_stream.isatty()

    def fileno(self):
        return self.terminal_stream.fileno()


class RunLoggingContext:
    def __init__(self, project_root, logging_root=None, run_name=None,
                 slack_header=None):
        self.project_root = Path(project_root).resolve()
        self.started_at = datetime.now(timezone(timedelta(hours=9)))

        root = Path(logging_root) if logging_root else self.project_root / "logging"
        self.run_dir = _allocate_run_directory(
            root,
            self.started_at,
            run_name=run_name,
        )
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir()

        self.monitor_path = self.run_dir / "monitor.sh"
        self.monitor_path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'cd "$(dirname "$0")"\n'
            "tail -n 100 -F train.log\n",
            encoding="utf-8",
        )
        self.monitor_path.chmod(0o755)

        self.log_path = self.run_dir / "train.log"
        self.args_path = self.run_dir / "args.json"
        self.config_path = self.run_dir / "config.json"
        self.metadata_path = self.run_dir / "run_metadata.json"

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._slack_notifier = SlackEpochNotifier(
            self.run_dir.name,
            header=slack_header,
        )
        sys.stdout = _TeeStream(
            self._stdout,
            self._log_handle,
            self._lock,
            self._slack_notifier,
        )
        sys.stderr = _TeeStream(
            self._stderr,
            self._log_handle,
            self._lock,
            self._slack_notifier,
        )
        print("Slack notifications: {}".format(
            "enabled" if self._slack_notifier.enabled else "disabled"))
        self._closed = False

        self._save_metadata(device=None)
        atexit.register(self.close)

    def _save_metadata(self, device=None):
        _write_json(
            self.metadata_path,
            {
                "started_at": self.started_at.isoformat(),
                "project_root": str(self.project_root),
                "run_dir": str(self.run_dir),
                "checkpoint_dir": str(self.checkpoint_dir),
                "log_path": str(self.log_path),
                "cwd": os.getcwd(),
                "command": " ".join(shlex.quote(argument) for argument in sys.argv),
                "argv": sys.argv,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "pid": os.getpid(),
                "device": str(device) if device is not None else None,
            },
        )

    def save_configuration(self, args, config, device=None):
        requested_checkpoint_path = getattr(args, "multi_cls_model_path", None)
        requested_log_file = getattr(args, "log_file", None)

        # 기존 trainer_ext.py의 저장 코드를 그대로 실행별 checkpoints로 연결한다.
        args.run_dir = str(self.run_dir)
        args.multi_cls_model_path = str(self.checkpoint_dir)
        args.log_file = str(self.log_path)
        args.console_capture_active = True
        args.requested_multi_cls_model_path = requested_checkpoint_path
        args.requested_log_file = requested_log_file

        _write_json(self.args_path, vars(args))
        _write_json(self.config_path, vars(config))
        self._save_metadata(device=device)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            if sys.stdout is not self._stdout:
                sys.stdout = self._stdout
            if sys.stderr is not self._stderr:
                sys.stderr = self._stderr
            self._log_handle.close()
            self._slack_notifier.close()


def start_run_logging(project_root, logging_root=None, run_name=None,
                      slack_header=None):
    return RunLoggingContext(
        project_root,
        logging_root=logging_root,
        run_name=run_name,
        slack_header=slack_header,
    )
