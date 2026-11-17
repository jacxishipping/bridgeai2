"""Shell execution, package installation and workflows (F-04-06 split of tools.py)."""
# Code in this module was moved verbatim from src/ai_bridge/tools.py
# (F-04-06 package split). Behaviour is unchanged; only the module
# location changed. Import through `ai_bridge.tools` for compatibility.

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import time

from ..policy import Policy
from ..security import confined_path, redact, workspace_root
from ..terminal_manager import terminate_process_group
from ._common import _check, default_policy
from .desktop import wait_for_text

_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+$")
_LINUX_PKG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def install_package(package_id: str, version: str | None = None, *, approval_granted: bool = False, policy: Policy | None = None) -> dict:
    """Install a package on a Windows agent (winget) or Linux/WSL agent (apt/dnf/pacman/apk).

    Tier-3 (sensitive): the control plane holds the task in
    needs_approval until a human approves it, and dispatch() passes
    that approval through here. Policy is enforced by dispatch() and
    re-checked here so direct calls stay gated too.
    The package id is validated and passed as a fixed argv list
    (shell=False) — no shell text from arguments is ever executed.
    """
    _check("install_package", approval_granted=approval_granted, policy=policy)
    if not isinstance(package_id, str) or not package_id.strip():
        raise ValueError("package_id must be a non-empty string.")
    package_id = package_id.strip()

    if version is not None and (not isinstance(version, str) or not _VERSION_RE.match(version)):
        raise ValueError("version must be letters/digits/dot/dash/underscore only.")

    current_os = platform.system()

    if current_os == "Windows":
        if not _PACKAGE_ID_RE.match(package_id):
            raise ValueError("package_id must look like a winget id (Publisher.App, letters/digits/dot/dash/underscore only).")
        if shutil.which("winget") is None:
            return {"available": False,
                    "note": "winget not found on this machine. Install App Installer (winget) from the Microsoft Store first."}
        argv = ["winget", "install", "--id", package_id, "-e",
                "--accept-source-agreements", "--accept-package-agreements"]
        if version:
            argv += ["--version", version]
        try:
            proc = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"winget install of {package_id} timed out after 900s.")
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()[:8000]
        if proc.returncode != 0:
            raise RuntimeError(f"winget install of {package_id} failed (exit {proc.returncode}): {output}")
        return {"installed": package_id, "version": version, "manager": "winget", "output": output}

    # Linux / WSL / Darwin
    pkg_name = package_id.split(".")[-1] if ("." in package_id and not _LINUX_PKG_RE.match(package_id)) else package_id
    if not _LINUX_PKG_RE.match(pkg_name):
        raise ValueError(f"Invalid package name {package_id!r}: letters, digits, dot, dash, underscore, plus only.")

    is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
    sudo_prefix = [] if is_root else (["sudo", "-n"] if shutil.which("sudo") else [])

    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"

    if shutil.which("apt-get"):
        target_spec = f"{pkg_name}={version}" if version else pkg_name
        argv = [*sudo_prefix, "apt-get", "install", "-y", target_spec]
        mgr = "apt-get"
    elif shutil.which("dnf"):
        target_spec = f"{pkg_name}-{version}" if version else pkg_name
        argv = [*sudo_prefix, "dnf", "install", "-y", target_spec]
        mgr = "dnf"
    elif shutil.which("yum"):
        target_spec = f"{pkg_name}-{version}" if version else pkg_name
        argv = [*sudo_prefix, "yum", "install", "-y", target_spec]
        mgr = "yum"
    elif shutil.which("pacman"):
        argv = [*sudo_prefix, "pacman", "-S", "--noconfirm", pkg_name]
        mgr = "pacman"
    elif shutil.which("apk"):
        argv = [*sudo_prefix, "apk", "add", pkg_name]
        mgr = "apk"
    else:
        return {"available": False, "note": "install_package found no supported package manager (apt-get, dnf, yum, pacman, apk)."}

    try:
        proc = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{mgr} install of {pkg_name} timed out after 900s.")
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()[:8000]
    if proc.returncode != 0:
        raise RuntimeError(f"{mgr} install of {pkg_name} failed (exit {proc.returncode}): {output}")
    return {"installed": pkg_name, "version": version, "manager": mgr, "output": output}


_SCRUB_ENV_KEYS = frozenset({
    "AI_BRIDGE_ADMIN_TOKEN",
    "AI_BRIDGE_DEVICE_TOKEN",
    "AI_BRIDGE_OIDC_CLIENT_SECRET",
})
# SEC-07 (Wave 2A): _SCRUB_ENV_KEYS above is retained only for reference —
# blocklisting three keys is not enough. _run_argv now builds a minimal
# allowlist env (like execution.py) so DATABASE_URL, vault/TURN/webhook/
# OAuth material and AI_BRIDGE_* secrets never reach a child process.

_CHILD_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM")
_CHILD_ENV_WINDOWS_ALLOWLIST = (
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "LOCALAPPDATA", "APPDATA",
)


def _clean_child_env() -> dict[str, str]:
    """Minimal allowlist env for child processes (SEC-07).

    Only PATH, HOME, LANG, LC_ALL, TZ, TERM are inherited (plus the Windows
    system vars a child needs to start). Everything else — DATABASE_URL,
    vault/TURN/webhook/OAuth tokens, AI_BRIDGE_* secrets — is dropped.
    """
    env: dict[str, str] = {}
    for key in _CHILD_ENV_ALLOWLIST:
        if key in os.environ:
            env[key] = os.environ[key]
    if "PATH" not in env:
        env["PATH"] = (
            r"C:\Windows\System32;C:\Windows" if os.name == "nt"
            else "/usr/local/bin:/usr/bin:/bin"
        )
    if "LANG" not in env:
        env["LANG"] = "C.UTF-8"
    if os.name == "nt":
        for key in _CHILD_ENV_WINDOWS_ALLOWLIST:
            if key in os.environ:
                env[key] = os.environ[key]
    return env


def _split_argv(command: str) -> list[str]:
    """Split a command line into argv without involving a shell."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string.")
    lexer = shlex.shlex(command, posix=(os.name != "nt"))
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        argv = list(lexer)
    except ValueError as exc:
        raise ValueError(f"command could not be parsed: {exc}") from exc
    if os.name == "nt":
        cleaned = []
        for token in argv:
            if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
                token = token[1:-1]
            cleaned.append(token)
        argv = cleaned
    return [a for a in argv if a]


def _run_cwd(working_directory: str, *, is_admin: bool, policy: Policy | None) -> str | None:
    if not working_directory:
        try:
            ws = workspace_root()
        except Exception:  # noqa: BLE001
            return None
        return str(ws) if ws.is_dir() else None
    try:
        return str(confined_path(working_directory))
    except ValueError:
        host_tools_enabled = os.environ.get("AI_BRIDGE_DISABLE_HOST_TOOLS", "1") == "0"
        if (host_tools_enabled or (policy and policy.unrestricted) or is_admin) and os.path.isabs(working_directory) and os.path.isdir(working_directory):
            return working_directory
        raise


def _terminate_proc_tree(proc: subprocess.Popen[str]) -> None:
    """Terminate *proc* and its process group/tree, draining pipes best-effort."""
    terminate_process_group(proc, grace_seconds=2.0)
    # Windows second-chance: terminate_process_group runs taskkill /F /T once
    # and returns without verifying; if the process survived, try once more.
    if os.name == "nt" and proc.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                shell=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    try:
        proc.communicate(timeout=5)
    except Exception:
        pass


def _run_argv(argv: list[str], cwd: str | None, timeout_arg: int) -> dict:
    from ..control.task_runtime import TaskCancelled, cooperative_sleep, raise_if_cancelled

    child_env = _clean_child_env()
    proc: subprocess.Popen[str] | None = None
    try:
        raise_if_cancelled()
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        deadline = time.time() + timeout_arg
        while proc.poll() is None:
            raise_if_cancelled()
            if time.time() >= deadline:
                # F-04-11: SIGTERM, grace, then SIGKILL the whole process group.
                _terminate_proc_tree(proc)
                return {
                    "returncode": None,
                    "timeout": True,
                    "stdout": "",
                    "stderr": f"command timed out after {timeout_arg}s",
                }
            cooperative_sleep(0.25, interval=0.25)
        out, err = proc.communicate(timeout=5)
        return {
            "returncode": proc.returncode,
            "stdout": redact((out or "")[-20000:]),
            "stderr": redact((err or "")[-20000:]),
        }
    except TaskCancelled:
        if proc is not None and proc.poll() is None:
            _terminate_proc_tree(proc)
        raise
    except FileNotFoundError:
        return {"returncode": None, "error": f"executable not found: {argv[0]}", "stdout": "", "stderr": ""}
    except Exception as exc:  # noqa: BLE001
        return {"returncode": None, "error": redact(str(exc))}


def run_shell(
    command: str,
    working_directory: str = "",
    timeout: int = 0,
    *,
    approval_granted: bool = False,
    is_admin: bool = False,
    policy: Policy | None = None,
) -> dict:
    """Run a command as an argv list (shell=False). Tier-3 sensitive tool.

    The line is split with shlex and the first token is the executable, so
    pipes, redirection, &&, globbing and variable expansion are NOT
    interpreted. Shell semantics exist only in `terminal_exec`, which is
    reserved for the control plane's interactive terminal relay.
    """
    _check("run_shell", approval_granted=approval_granted, is_admin=is_admin, policy=policy)
    argv = _split_argv(command)
    timeout_arg = min(int(timeout), 3600) if timeout and int(timeout) > 0 else 300
    return _run_argv(argv, _run_cwd(working_directory, is_admin=is_admin, policy=policy), timeout_arg)


def terminal_exec(
    command: str,
    working_directory: str = "",
    timeout: int = 0,
    *,
    approval_granted: bool = False,
    is_admin: bool = False,
    policy: Policy | None = None,
) -> dict:
    """Run a command through the platform shell in argv form (never shell=True).

    Only the control plane's live-terminal relay calls this so typed shell
    commands behave as a shell user expects. It is not an MCP tool.
    """
    _check("terminal_exec", approval_granted=approval_granted, is_admin=is_admin, policy=policy)
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string.")
    timeout_arg = min(int(timeout), 3600) if timeout and int(timeout) > 0 else 300
    if os.name == "nt":
        argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        shell = os.environ.get("SHELL", "/bin/bash")
        if not os.path.exists(shell):
            shell = "/bin/sh"
        argv = [shell, "-c", command]
    return _run_argv(argv, _run_cwd(working_directory, is_admin=is_admin, policy=policy), timeout_arg)


def execute_workflow(
    steps: list[dict],
    *,
    stop_on_failure: bool = True,
    max_retries: int = 1,
    approval_granted: bool = False,
    policy: Policy | None = None,
) -> dict:
    """Execute a structured plan of autonomous actions sequentially with verification and retry."""
    eff_policy = policy or default_policy()
    _check("execute_workflow", approval_granted=approval_granted, policy=eff_policy)
    from ..control.dispatch import dispatch

    results = []
    workflow_ok = True
    start_total = time.time()

    for idx, step in enumerate(steps or []):
        step_name = step.get("name") or f"Step {idx + 1}"
        action = step.get("action")
        args = dict(step.get("args") or {})
        verify_cfg = step.get("verify")
        retries_left = max(0, int(step.get("retries", max_retries)))

        step_res = {
            "index": idx + 1,
            "name": step_name,
            "action": action,
            "args": args,
            "status": "pending",
            "duration_seconds": 0.0,
            "result": None,
            "error": None,
        }

        t0 = time.time()
        step_success = False

        while retries_left >= 0 and not step_success:
            try:
                act_res = dispatch(action, args, policy=eff_policy, approval_granted=approval_granted)
                step_res["result"] = act_res

                if verify_cfg and isinstance(verify_cfg, dict):
                    v_text = verify_cfg.get("text")
                    v_timeout = float(verify_cfg.get("timeout_seconds", 8.0))
                    if v_text:
                        v_res = wait_for_text(v_text, timeout_seconds=v_timeout, approval_granted=approval_granted, policy=eff_policy)
                        step_res["verification"] = v_res
                        if not v_res.get("found"):
                            raise RuntimeError(f"Step verification failed: text '{v_text}' not found after {v_timeout}s.")

                step_res["status"] = "completed"
                step_success = True
            except Exception as exc:
                retries_left -= 1
                if retries_left < 0:
                    step_res["status"] = "failed"
                    step_res["error"] = str(exc)
                    workflow_ok = False
                else:
                    time.sleep(1.0)

        step_res["duration_seconds"] = round(time.time() - t0, 2)
        results.append(step_res)

        if not step_success and stop_on_failure:
            break

    total_duration = round(time.time() - start_total, 2)
    return {
        "ok": workflow_ok,
        "total_steps": len(steps or []),
        "executed_steps": len(results),
        "successful_steps": sum(1 for r in results if r["status"] == "completed"),
        "failed_steps": sum(1 for r in results if r["status"] == "failed"),
        "total_duration_seconds": total_duration,
        "steps": results,
    }
