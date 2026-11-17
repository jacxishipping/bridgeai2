"""Wave 3C Track B — Python-side tests for F13-06..F13-10 and F08-09 (Python half).

* F13-06: simulate_agent_intent is OFF by default, on only with
  AI_BRIDGE_DEV_SIMULATE_INTENT=1.
* F13-07: RTCP-based adaptive bitrate is honest — real receiver reports
  via sender.getStats() are labelled "rtcp", viewer reports are labelled
  "viewer_reported", and with no stats at all nothing is fabricated
  (abr_source "none", abr_available False).
* F13-08: mouse buttons held in the fallback (non-DLL) path are tracked
  and released on disconnect / session teardown (Wave 1B key release,
  extended to buttons).
* F13-09: stale sessions are reconciled at start — a restarted
  server/agent refuses to resume a previous incarnation on a recycled
  session id; a fresh session gets a fresh id.
* F13-10: no offer is posted until ICE gathering is complete
  (wait_for_ice_gathering_complete gate).
* F08-09 (Python): one secure-desktop rule — any non-"Default" desktop
  name is secure.

No aiortc / live network is needed: AgentStreamer is exercised
directly and the peer connection is faked where required.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from ai_bridge import screen_capture, webrtc_stream
from ai_bridge.webrtc_stream import (
    AgentStreamer,
    DictSignaling,
    describe_adaptive_bitrate,
    dev_simulate_intent_enabled,
    get_held_mouse_buttons,
    is_stale_session_state,
    reconcile_session_before_start,
    release_all_held_keys,
    start_stream,
    stop_stream,
    track_mouse_button_state,
    wait_for_ice_gathering_complete,
)


@pytest.fixture(autouse=True)
def _clear_global_input_state(monkeypatch):
    """Reset process-global input gates / held state so each test measures
    its own session, not leftover state from other modules."""
    from ai_bridge.plane_control_flags import set_emergency_stop

    set_emergency_stop(False)
    monkeypatch.delenv("AI_BRIDGE_DEV_SIMULATE_INTENT", raising=False)
    try:
        from ai_bridge.control.v1_routes import _HUMAN_TAKEOVER_BY_DEVICE

        _HUMAN_TAKEOVER_BY_DEVICE.clear()
    except Exception:
        pass
    release_all_held_keys()
    yield
    set_emergency_stop(False)
    release_all_held_keys()
    try:
        from ai_bridge.control.v1_routes import _HUMAN_TAKEOVER_BY_DEVICE

        _HUMAN_TAKEOVER_BY_DEVICE.clear()
    except Exception:
        pass
    webrtc_stream._STREAM_TELEMETRY.clear()


def _human_control_session(**extra):
    sess = {
        "mode": "human_control",
        "permissions": {"control_mouse": True, "control_keyboard": True},
        "status": "waiting",
        "created_at": time.time(),
    }
    sess.update(extra)
    return sess


def _streamer(session_id="sess-phase3", **extra):
    return AgentStreamer(session_id, DictSignaling(_human_control_session(**extra)))


class _FakeChannel:
    def __init__(self):
        self.readyState = "open"
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)


class _FakeSignaling:
    """Stand-in for HttpSignaling: no .session attribute; pollable."""

    def __init__(self, polled=None, poll_raises=False):
        self.polled = dict(polled or {"status": "waiting", "created_at": time.time()})
        self.poll_raises = poll_raises
        self.errors = []

    def poll_session(self):
        if self.poll_raises:
            raise RuntimeError("network down")
        return dict(self.polled)

    def post_offer(self, offer):
        pass

    def post_candidate(self, candidate):
        pass

    def post_error(self, message):
        self.errors.append(message)


# --- F13-06: simulate_agent_intent gated off by default ----------------------


def test_simulate_intent_disabled_by_default(monkeypatch):
    assert dev_simulate_intent_enabled() is False
    streamer = _streamer()
    channel = _FakeChannel()
    streamer.input_channel = channel
    res = streamer._handle_input_payload(json.dumps({"action": "simulate_agent_intent"}))
    assert res == {"ok": False, "error": "simulate_agent_intent_disabled"}
    assert channel.sent == []


def test_simulate_intent_enabled_only_with_explicit_dev_flag(monkeypatch):
    monkeypatch.setenv("AI_BRIDGE_DEV_SIMULATE_INTENT", "1")
    assert dev_simulate_intent_enabled() is True
    streamer = _streamer()
    channel = _FakeChannel()
    streamer.input_channel = channel
    res = streamer._handle_input_payload(json.dumps({"action": "simulate_agent_intent"}))
    assert res["ok"] is True
    assert res["action"] == "simulated_intent"
    assert len(channel.sent) == 1
    assert json.loads(channel.sent[0])["type"] == "agent_intent"


def test_simulate_intent_flag_other_values_stay_off(monkeypatch):
    for value in ("0", "true", "yes", "on"):
        monkeypatch.setenv("AI_BRIDGE_DEV_SIMULATE_INTENT", value)
        assert dev_simulate_intent_enabled() is False


# --- F13-07: honest RTCP-based adaptive bitrate ------------------------------


def test_describe_adaptive_bitrate_is_honest():
    desc = describe_adaptive_bitrate()
    assert desc["mechanism"] == "rtcp_receiver_reports_via_aiortc_sender_getStats"
    assert desc["encoder_reconfigured"] is False
    assert "none" in desc["abr_sources"]


def test_apply_rtcp_feedback_labels_source_and_updates_telemetry():
    streamer = _streamer("sess-abr")
    webrtc_stream._STREAM_TELEMETRY["sess-abr"] = {}
    before = streamer.bitrate_kbps
    res = streamer.apply_rtcp_feedback(300.0, 10.0, source="rtcp")
    assert res["abr_source"] == "rtcp"
    assert res["bitrate_kbps"] < before  # heavy loss steps bitrate down
    telem = webrtc_stream._STREAM_TELEMETRY["sess-abr"]
    assert telem["abr_source"] == "rtcp"
    assert telem["abr_available"] is True
    assert telem["rtt_ms"] == 300.0
    assert telem["packet_loss_pct"] == 10.0


def test_network_quality_viewer_report_is_labelled_not_rtcp():
    streamer = _streamer("sess-nq")
    webrtc_stream._STREAM_TELEMETRY["sess-nq"] = {}
    res = streamer._handle_input_payload(
        json.dumps({"action": "network_quality", "rtt_ms": 300, "packet_loss_pct": 10})
    )
    assert res["ok"] is True
    assert res["abr_source"] == "viewer_reported"
    assert webrtc_stream._STREAM_TELEMETRY["sess-nq"]["abr_source"] == "viewer_reported"
    assert webrtc_stream._STREAM_TELEMETRY["sess-nq"]["abr_available"] is False


def test_poll_rtcp_stats_no_sender_reports_nothing_fabricated():
    streamer = _streamer("sess-rtcp-none")
    webrtc_stream._STREAM_TELEMETRY["sess-rtcp-none"] = {}

    class _PcNoSender:
        def getSenders(self):
            return []

    streamer.pc = _PcNoSender()
    res = asyncio.run(streamer._poll_rtcp_stats())
    assert res is None
    assert streamer._rtcp_available is False
    telem = webrtc_stream._STREAM_TELEMETRY["sess-rtcp-none"]
    assert telem["abr_source"] == "none"
    assert telem["abr_available"] is False
    assert "rtt_ms" not in telem


def test_poll_rtcp_stats_uses_real_receiver_report():
    streamer = _streamer("sess-rtcp")
    webrtc_stream._STREAM_TELEMETRY["sess-rtcp"] = {}
    track = object()
    streamer.video_track = track

    class _Sender:
        def __init__(self):
            self.track = track

        async def getStats(self):
            return {
                "remote": SimpleNamespace(
                    type="remote-inbound-rtp",
                    roundTripTime=0.2,
                    packetsLost=10,
                    packetsSent=90,
                )
            }

    class _Pc:
        def getSenders(self):
            return [_Sender()]

    streamer.pc = _Pc()
    before = streamer.bitrate_kbps
    res = asyncio.run(streamer._poll_rtcp_stats())
    assert res is not None
    assert res["abr_source"] == "rtcp"
    assert streamer._rtcp_available is True
    assert res["bitrate_kbps"] < before  # 10% loss steps down
    telem = webrtc_stream._STREAM_TELEMETRY["sess-rtcp"]
    assert telem["rtt_ms"] == 200.0
    assert telem["packet_loss_pct"] == 10.0


# --- F13-08: mouse-button release in the fallback path -----------------------


def test_mouse_button_track_and_release():
    track_mouse_button_state("left", True)
    track_mouse_button_state("right", True)
    assert get_held_mouse_buttons() == ["left", "right"]
    released = release_all_held_keys()
    assert released >= 0
    assert get_held_mouse_buttons() == []
    track_mouse_button_state("left", True)
    track_mouse_button_state("left", False)
    assert get_held_mouse_buttons() == []


def test_mousedown_via_input_handler_is_tracked_and_released_on_teardown():
    streamer = _streamer("sess-mouse")
    # On this (non-Windows, no-DLL) host the fallback input path runs;
    # the button must be tracked even though no native DLL holds it.
    res = streamer._handle_input_payload(
        json.dumps({"action": "mousedown", "button": "left", "x": 0.5, "y": 0.5})
    )
    assert get_held_mouse_buttons() == ["left"]
    # Session teardown (stop) releases held buttons as well as keys.
    streamer.stop()
    assert get_held_mouse_buttons() == []
    assert res is not None  # result may be ok or fallback-error; tracking is what matters


def test_mouseup_via_input_handler_untracks():
    streamer = _streamer("sess-mouse-up")
    streamer._handle_input_payload(
        json.dumps({"action": "mousedown", "button": "right", "x": 0.5, "y": 0.5})
    )
    assert get_held_mouse_buttons() == ["right"]
    streamer._handle_input_payload(
        json.dumps({"action": "mouseup", "button": "right", "x": 0.5, "y": 0.5})
    )
    assert get_held_mouse_buttons() == []


def test_stop_stream_releases_held_buttons():
    track_mouse_button_state("middle", True)
    assert get_held_mouse_buttons() == ["middle"]
    stop_stream("sess-never-started")
    assert get_held_mouse_buttons() == []


# --- F13-09: session reconciliation after restart ----------------------------


def test_is_stale_session_state():
    now = time.time()
    assert is_stale_session_state({"status": "waiting", "created_at": now}) == (False, "")
    assert is_stale_session_state({"status": "ended"})[0] is True
    assert is_stale_session_state({"status": "error"})[0] is True
    assert is_stale_session_state({"status": "waiting", "created_at": now - 601})[0] is True
    assert is_stale_session_state({"status": "waiting", "created_at": now - 599}) == (False, "")
    # Previous incarnation already negotiated offer+answer.
    assert is_stale_session_state(
        {"status": "answered", "offer": {"sdp": "v=0"}, "answer": {"sdp": "v=0"}}
    )[0] is True
    # Answered but no answer yet: not positive evidence of staleness.
    assert is_stale_session_state({"status": "answered", "offer": {"sdp": "v=0"}}) == (False, "")
    # Unknown / empty state is not stale (transient poll failure must not kill a session).
    assert is_stale_session_state({}) == (False, "")
    assert is_stale_session_state(None) == (False, "")


def test_reconcile_fresh_dict_session_ok():
    rec = reconcile_session_before_start(
        DictSignaling(_human_control_session()), "sess-fresh"
    )
    assert rec == {"ok": True, "stale": False, "reason": "", "session_id": "sess-fresh"}


def test_reconcile_stale_dict_session_refused():
    rec = reconcile_session_before_start(
        DictSignaling(
            _human_control_session(
                status="answered", offer={"sdp": "v=0"}, answer={"sdp": "v=0"}
            )
        ),
        "sess-old",
    )
    assert rec["ok"] is False
    assert rec["stale"] is True
    assert rec["reason"] == "session_already_negotiated"


def test_reconcile_remote_poll_stale_refused_and_poll_failure_not_stale():
    stale_sig = _FakeSignaling({"status": "ended", "created_at": time.time()})
    rec = reconcile_session_before_start(stale_sig, "sess-remote-old")
    assert rec["ok"] is False and rec["reason"] == "session_ended"
    down_sig = _FakeSignaling(poll_raises=True)
    rec = reconcile_session_before_start(down_sig, "sess-remote-down")
    assert rec["ok"] is True  # poll failure is unknown, not stale


def test_start_stream_refuses_stale_session_id():
    stale = DictSignaling(
        _human_control_session(
            status="connected", offer={"sdp": "v=0"}, answer={"sdp": "v=0"}
        )
    )
    with pytest.raises(RuntimeError, match="Stale WebRTC session"):
        start_stream("sess-stale-id", stale)
    assert "sess-stale-id" not in webrtc_stream._STREAMERS


def test_start_stream_fresh_session_starts():
    fresh = DictSignaling(_human_control_session())
    res = start_stream("sess-fresh-id", fresh)
    assert res["status"] == "started"
    assert res["session_id"] == "sess-fresh-id"
    # The streamer thread fails fast here (aiortc is not installed in
    # this environment) and unregisters itself; stop must not raise.
    time.sleep(0.3)
    stop_stream("sess-fresh-id")


# --- F13-10: ICE-complete before offer ---------------------------------------


class _FakePc:
    def __init__(self, state="new"):
        self.iceGatheringState = state


def test_wait_ice_gathering_complete_immediately():
    assert asyncio.run(wait_for_ice_gathering_complete(_FakePc("complete"), timeout_seconds=1)) is True


def test_wait_ice_gathering_times_out_without_complete():
    assert (
        asyncio.run(wait_for_ice_gathering_complete(_FakePc("gathering"), timeout_seconds=0.1))
        is False
    )


def test_wait_ice_gathering_becomes_complete():
    pc = _FakePc("gathering")

    async def _flip():
        await asyncio.sleep(0.15)
        pc.iceGatheringState = "complete"

    async def _main():
        task = asyncio.create_task(_flip())
        ok = await wait_for_ice_gathering_complete(pc, timeout_seconds=2)
        await task
        return ok

    assert asyncio.run(_main()) is True


# --- F08-09 (Python): one secure-desktop rule --------------------------------


def test_secure_desktop_name_rule():
    assert screen_capture.is_secure_desktop_name("Default") is False
    assert screen_capture.is_secure_desktop_name("default") is False
    assert screen_capture.is_secure_desktop_name("Winlogon") is True
    assert screen_capture.is_secure_desktop_name("Disconnect") is True
    # The old Python list missed the hyphenated name; the unified
    # (non-"Default") rule catches it, matching the C++ helper.
    assert screen_capture.is_secure_desktop_name("Screen-saver") is True
    assert screen_capture.is_secure_desktop_name("") is False


def test_legacy_webrtc_session_error_preserves_deprecation_headers(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from ai_bridge.control.app import create_app

    app = create_app(admin_token="adm-dep", db_path=tmp_path / "dep.db")
    client = TestClient(app)
    headers = {"x-admin-token": "adm-dep"}

    monkeypatch.setenv("AI_BRIDGE_PROFILE", "production")
    monkeypatch.setenv("AI_BRIDGE_CSRF_SECRET", "test-csrf-secret")
    monkeypatch.delenv("AI_BRIDGE_ALLOW_LEGACY_WEBRTC", raising=False)
    r410 = client.post("/api/devices/local/desktop/webrtc/session", headers=headers, json={"fps": 15})
    assert r410.status_code == 410
    assert r410.headers.get("deprecation") == "true"
    assert "v1/sessions" in (r410.headers.get("link") or "")
    assert r410.headers.get("sunset") == "legacy-webrtc-session-route"

    monkeypatch.setenv("AI_BRIDGE_ALLOW_LEGACY_WEBRTC", "1")
    r404 = client.post("/api/devices/nonexistent-dev/desktop/webrtc/session", headers=headers, json={"fps": 15})
    assert r404.status_code == 404
    assert r404.headers.get("deprecation") == "true"
    assert "v1/sessions" in (r404.headers.get("link") or "")
    assert r404.headers.get("sunset") == "legacy-webrtc-session-route"
