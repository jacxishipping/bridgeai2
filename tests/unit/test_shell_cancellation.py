"""Verify tools.shell._run_argv terminates its child process on TaskCancelled."""
from __future__ import annotations

import subprocess
import sys

import pytest

from ai_bridge.control.task_runtime import TaskCancelled
from ai_bridge.tools import shell


def test_run_argv_kills_child_on_task_cancelled(monkeypatch):
    spawned: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def _recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if args and isinstance(args[0], (list, tuple)) and args[0] and args[0][0] == sys.executable:
            spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)

    calls = {"n": 0}

    def _cancel_on_second_check():
        calls["n"] += 1
        if calls["n"] >= 2:
            raise TaskCancelled("cancelled in test")

    monkeypatch.setattr("ai_bridge.control.task_runtime.raise_if_cancelled", _cancel_on_second_check)

    with pytest.raises(TaskCancelled):
        shell._run_argv([sys.executable, "-c", "import time; time.sleep(30)"], None, 30)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
