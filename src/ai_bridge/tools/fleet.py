"""Fleet commands, frame capture and agent updates (F-04-06 split of tools.py)."""
# Code in this module was moved verbatim from src/ai_bridge/tools.py
# (F-04-06 package split). Behaviour is unchanged; only the module
# location changed. Import through `ai_bridge.tools` for compatibility.

from __future__ import annotations

import base64
import io
import os
import platform
import time
from pathlib import Path

from ..policy import Policy
from . import _common  # LAST_FRAME_SCALE lives here (shared mutable state)
from ._common import _check, default_policy
from .desktop import _grab_windows_desktop_fallback


def broadcast_fleet_command(
    tool: str,
    args: dict | None = None,
    target_os: str | None = None,
    target_device_ids: list[str] | None = None,
    approval_granted: bool = False,
    policy: Policy | None = None,
) -> dict:
    """Broadcast an action or diagnostic tool across multiple fleet devices simultaneously."""
    eff_policy = policy or default_policy()
    _check("broadcast_fleet_command", approval_granted=approval_granted, policy=eff_policy)
    from ..control.dispatch import dispatch

    local_res = dispatch(tool, args or {}, policy=eff_policy, approval_granted=approval_granted)
    return {
        "ok": True,
        "batch_id": "local-broadcast",
        "device_count": 1,
        "results": [{"device_id": "local", "status": "completed", "result": local_res}],
    }


def capture_desktop_frame(
    scale: float = 0.75,
    quality: int = 70,
    *,
    approval_granted: bool = False,
    policy: Policy | None = None,
) -> dict:
    """Capture a lightweight scaled desktop frame optimized for real-time visual streaming."""
    eff_policy = policy or default_policy()
    _check("capture_desktop_frame", approval_granted=approval_granted, policy=eff_policy)

    try:
        from PIL import Image, ImageGrab
    except Exception as exc:
        return {
            "available": False,
            "width": 0,
            "height": 0,
            "orig_width": 0,
            "orig_height": 0,
            "mime_type": "image/jpeg",
            "size_bytes": 0,
            "base64_data": "",
            "timestamp": time.time(),
            "note": f"Desktop capture requires Pillow: {exc}",
        }

    try:
        shot = ImageGrab.grab()
    except Exception as exc:
        if platform.system() == "Windows":
            shot = _grab_windows_desktop_fallback()
        else:
            return {
                "available": False,
                "width": 0,
                "height": 0,
                "orig_width": 0,
                "orig_height": 0,
                "mime_type": "image/jpeg",
                "size_bytes": 0,
                "base64_data": "",
                "timestamp": time.time(),
                "note": f"Desktop capture unavailable on headless host: {exc}",
            }

    orig_w, orig_h = shot.size
    scale = max(0.2, min(1.0, float(scale)))
    quality = max(20, min(95, int(quality)))

    if scale < 0.99:
        new_w = max(100, int(orig_w * scale))
        new_h = max(100, int(orig_h * scale))
        shot = shot.resize((new_w, new_h), Image.Resampling.BILINEAR)
    else:
        new_w, new_h = orig_w, orig_h

    if shot.mode in ("RGBA", "P"):
        shot = shot.convert("RGB")

    buf = io.BytesIO()
    shot.save(buf, format="JPEG", quality=quality)
    jpg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpg_bytes).decode("ascii")

    _common.LAST_FRAME_SCALE = (new_w / orig_w) if orig_w else 1.0
    return {
        "available": True,
        "width": new_w,
        "height": new_h,
        "image_width": new_w,
        "image_height": new_h,
        "orig_width": orig_w,
        "orig_height": orig_h,
        "scale": scale,
        "quality": quality,
        "mime_type": "image/jpeg",
        "size_bytes": len(jpg_bytes),
        "base64_data": b64,
        "timestamp": time.time(),
    }


def update_fleet_agent(
    *,
    approval_granted: bool = False,
    policy: Policy | None = None,
    repo_dir: str | Path | None = None,
) -> dict:
    """Self-update the local agent to a SIGNED, pinned release commit.

    R-P0-16 (Wave 2A): the old unsigned ``git fetch origin main &&
    git checkout -f -B main origin/main`` path is deleted. A signed release
    pin is required — manifest + signature + public key via
    ``AI_BRIDGE_UPDATE_MANIFEST_PATH`` / ``AI_BRIDGE_UPDATE_SIGNATURE_PATH``
    / ``AI_BRIDGE_UPDATE_PUBLIC_KEY`` (see ``ai_bridge.updates``). If the
    signed update is not configured this fails closed with PermissionError
    and performs NO checkout. ``repo_dir`` defaults to the current working
    directory (the checkout root); the control plane passes it explicitly
    when the repo lives elsewhere.
    """
    eff_policy = policy or default_policy()
    _check("update_fleet_agent", approval_granted=approval_granted, policy=eff_policy)
    from .. import updates

    if not updates.signed_update_configured():
        raise PermissionError(
            "Refusing unsigned fleet update: a signed release pin is required. "
            "Set AI_BRIDGE_UPDATE_MANIFEST_PATH, AI_BRIDGE_UPDATE_SIGNATURE_PATH "
            "and AI_BRIDGE_UPDATE_PUBLIC_KEY to a signed release manifest "
            "(see ai_bridge.updates); there is no unsigned fallback."
        )
    manifest, signature_b64 = updates.load_signed_manifest()
    public_key_path = Path(os.environ[updates.UPDATE_PUBLIC_KEY_ENV])
    if repo_dir:
        target_repo = Path(repo_dir)
    else:
        env_repo = os.environ.get("AI_BRIDGE_REPO_DIR", "").strip()
        if env_repo:
            target_repo = Path(env_repo)
        else:
            pkg_root = Path(__file__).resolve().parents[3]
            target_repo = pkg_root if (pkg_root / ".git").exists() else Path.cwd()
    result = updates.apply_pinned_git_update(
        target_repo,
        manifest, signature_b64, public_key_path,
    )
    return {
        "ok": True,
        "returncode": 0,
        "stdout": result["commit"],
        "stderr": "",
        "timestamp": time.time(),
    }
