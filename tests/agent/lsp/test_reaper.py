"""Lifecycle tests for the LSP idle reaper and concurrency caps.

Background: on 2026-09-08 a ``pyright-langserver`` child of the Hermes
gateway survived ~98 hours idle with 2.1 GB RSS + 2.3 GB swap because
``idle_timeout`` / ``_last_used`` were recorded but never consumed.
These tests pin the contract that replaced that:

  * a client with an active lease is never reaped;
  * an idle client past ``idle_timeout`` is reaped, gracefully first;
  * using a client resets its idle clock;
  * a server that ignores shutdown/exit/SIGTERM is SIGKILLed within
    the bounded grace period;
  * the reaper cannot race a concurrent use, cannot reap twice, and
    repeated passes are idempotent;
  * caps evict the LRU *idle* client and refuse (never kill) when all
    clients are busy;
  * reaped keys are fully removed and a later use re-spawns;
  * ``shutdown()`` stops the reaper and tears every client down;
  * regression: with the automatic reaper an idle pyright cannot
    outlive ``idle_timeout`` + one reap interval.

Every test drives the real ``LSPService`` + ``LSPClient`` against the
stdlib-only mock server in ``_mock_lsp_server.py``; nothing is mocked
at the process level, so "reaped" means the OS pid is gone.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from agent.lsp import eventlog
from agent.lsp.manager import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MAX_SERVERS,
    DEFAULT_REAP_INTERVAL,
    LSPService,
)
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")

# The reaper samples /proc and the mock ignores SIGTERM via signal(); both
# are POSIX-only, matching the Linux gateway this component runs on.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only lifecycle tests")


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _swap_server(server_id: str, script: str, extra_env: dict | None = None):
    """Replace one registry entry with the mock; returns a restore fn."""
    idx = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[idx]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        if extra_env:
            env.update(extra_env)
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    SERVERS[idx] = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )

    def _restore():
        SERVERS[idx] = original

    return _restore


def _make_repo(base: Path, name: str, ext: str = ".py") -> Path:
    repo = base / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / ("x" + ext)).write_text("print('hi')\n" if ext == ".py" else "x\n")
    return repo


@pytest.fixture
def mock_registry(monkeypatch, tmp_path):
    """pyright + typescript both backed by the ``clean`` mock."""
    eventlog.reset_announce_caches()
    restores = [_swap_server("pyright", "clean"), _swap_server("typescript", "clean")]
    monkeypatch.chdir(str(tmp_path))
    yield tmp_path
    for r in restores:
        r()


@pytest.fixture
def stuck_registry(monkeypatch, tmp_path):
    """pyright backed by a server that ignores shutdown/exit/SIGTERM."""
    eventlog.reset_announce_caches()
    restore = _swap_server("pyright", "stuck")
    monkeypatch.chdir(str(tmp_path))
    yield tmp_path
    restore()


def _svc(**kw) -> LSPService:
    base = dict(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=3600.0,
        reap_interval=3600.0,
        shutdown_grace=5.0,
        max_servers=0,
        max_servers_per_id=0,
        start_reaper=False,
    )
    base.update(kw)
    return LSPService(**base)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Zombies still answer kill(0); the client awaits proc.wait() on
    # teardown so a reaped child is fully collected.
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("State:"):
                    return "Z" not in line.split()[1]
    except OSError:
        return False
    return True


def _wait_until(pred, timeout: float, step: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


def _mock_pids_for_root(root: Path) -> set:
    """Every live mock-server pid whose cwd is ``root`` — including any
    orphan the service no longer tracks.  /proc scan, no psutil."""
    out = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if os.readlink(p / "cwd") != str(root):
                continue
            cmd = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if MOCK_SERVER.encode() in cmd and _pid_alive(int(p.name)):
            out.add(int(p.name))
    return out


def _single_client(svc: LSPService):
    st = svc.get_status()
    assert st["client_count"] == 1, st
    return st["clients"][0]


# ---------------------------------------------------------------------------
# 1. active client is not reaped
# ---------------------------------------------------------------------------


def test_active_client_is_not_reaped(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.01)
    try:
        svc.get_diagnostics_sync(f)
        key = ("pyright", str(repo))
        assert key in svc._clients
        # Hold a lease exactly the way a caller does, then reap.
        with svc._state_lock:
            svc._inflight[key] = 1
        time.sleep(0.05)
        assert svc.reap_now() == 0
        assert key in svc._clients
        assert svc.get_status()["lifecycle"]["reaped_total"] == 0
        # Release the lease: now it is fair game.
        with svc._state_lock:
            svc._inflight.pop(key)
        time.sleep(0.05)
        assert svc.reap_now() == 1
        assert key not in svc._clients
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 2. idle client beyond timeout is reaped
# ---------------------------------------------------------------------------


def test_idle_client_beyond_timeout_is_reaped(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.2)
    try:
        svc.get_diagnostics_sync(f)
        pid = _single_client(svc)["pid"]
        assert pid and _pid_alive(pid)
        # Not idle long enough yet.
        assert svc.reap_now() == 0
        time.sleep(0.25)
        assert svc.reap_now() == 1
        assert svc.get_status()["client_count"] == 0
        assert _wait_until(lambda: not _pid_alive(pid), timeout=5.0)
        lc = svc.get_status()["lifecycle"]
        assert lc["reaped_total"] == 1
        assert lc["last_reap_at"] is not None
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 3. recently-used client resets its idle clock
# ---------------------------------------------------------------------------


def test_recent_use_resets_idle_clock(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    # Generous windows: each get_diagnostics_sync round trip costs
    # 0.1-0.3s on a loaded box, and the assertion is about ordering,
    # not precision (>1s of slack on every edge).
    svc = _svc(idle_timeout=3.0)
    try:
        svc.get_diagnostics_sync(f)
        first_pid = _single_client(svc)["pid"]
        time.sleep(1.5)
        svc.get_diagnostics_sync(f)          # touch: clock restarts
        touched = time.time()
        time.sleep(1.5)                       # ~3.0s+ since spawn, ~1.5s since use
        assert svc.reap_now() == 0, "reaped despite recent use"
        assert _single_client(svc)["pid"] == first_pid
        assert _single_client(svc)["idle_seconds"] < 3.0
        _wait_until(lambda: time.time() - touched > 3.2, timeout=5.0)
        assert svc.reap_now() == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 4. graceful shutdown path
# ---------------------------------------------------------------------------


def test_reap_uses_graceful_shutdown(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.05)
    try:
        svc.get_diagnostics_sync(f)
        pid = _single_client(svc)["pid"]
        time.sleep(0.1)
        t0 = time.time()
        assert svc.reap_now() == 1
        # The clean mock answers shutdown/exit immediately: no grace
        # timeout burned, nothing forced.
        assert time.time() - t0 < 3.0
        lc = svc.get_status()["lifecycle"]
        assert lc["forced_total"] == 0
        assert lc["graceful_failures"] == 0
        assert _wait_until(lambda: not _pid_alive(pid), timeout=5.0)
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 5. forced termination after timeout
# ---------------------------------------------------------------------------


def test_forced_termination_when_server_ignores_shutdown(stuck_registry):
    repo = _make_repo(stuck_registry, "r1")
    f = str(repo / "x.py")
    # Grace shorter than the client's own 2s shutdown-request timeout
    # so the *reaper's* bound is what fires.
    svc = _svc(idle_timeout=0.05, shutdown_grace=0.5)
    try:
        svc.get_diagnostics_sync(f)
        pid = _single_client(svc)["pid"]
        assert _pid_alive(pid)
        time.sleep(0.1)
        t0 = time.time()
        assert svc.reap_now() == 1
        elapsed = time.time() - t0
        # Hard bound: grace (0.5s) + SIGKILL + collection, nowhere near
        # the client's own 2s shutdown-request + 1s SIGTERM budget.
        assert elapsed < 2.5, f"forced path took {elapsed:.1f}s (grace was 0.5s)"
        assert _wait_until(lambda: not _pid_alive(pid), timeout=5.0), "SIGTERM-ignoring server survived"
        lc = svc.get_status()["lifecycle"]
        assert lc["reaped_total"] == 1
        assert lc["forced_total"] == 1
        assert lc["graceful_failures"] == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 6. concurrent reaper / use race
# ---------------------------------------------------------------------------


def test_reaper_cannot_race_concurrent_use(mock_registry):
    """Hammer the client from a worker while the reaper runs with a
    zero idle timeout.  Every use must complete without error, the
    service must never hand out a dead client, and at the end every
    pid the service ever spawned that is no longer registered must be
    gone (no orphans)."""
    import threading

    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.0001, reap_interval=1.0)
    pids_seen = set()
    errors = []
    stop = threading.Event()

    def _user():
        try:
            while not stop.is_set():
                diags = svc.get_diagnostics_sync(f)
                assert isinstance(diags, list)
                st = svc.get_status()
                for c in st["clients"]:
                    if c["pid"]:
                        pids_seen.add(c["pid"])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        workers = [threading.Thread(target=_user, daemon=True) for _ in range(3)]
        for w in workers:
            w.start()
        deadline = time.time() + 2.0
        reaps = 0
        while time.time() < deadline:
            reaps += svc.reap_now()
            time.sleep(0.02)
        stop.set()
        for w in workers:
            w.join(timeout=10)
        assert not errors, errors
        assert reaps >= 1, "reaper never found an idle window (test not exercising the race)"
        live = {c["pid"] for c in svc.get_status()["clients"]}
        # Every pid we ever saw that is not currently registered must have exited.
        for pid in pids_seen - live:
            assert _wait_until(lambda p=pid: not _pid_alive(p), timeout=5.0), f"orphan pid {pid}"
        # No key stuck in the reaping set.
        assert not svc._reaping
        assert not svc._inflight
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 7. repeated reaper cycles are idempotent / no double reap
# ---------------------------------------------------------------------------


def test_repeated_reaper_cycles_are_idempotent(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.05)
    try:
        for _ in range(3):
            assert svc.reap_now() == 0          # nothing to do, no side effects
        svc.get_diagnostics_sync(f)
        time.sleep(0.1)
        # Two passes launched together: exactly one may claim the client.
        async def _both():
            return await asyncio.gather(svc._reap_idle(), svc._reap_idle())

        results = svc._loop.run(_both(), timeout=15.0)
        assert sorted(results) == [0, 1]
        for _ in range(3):
            assert svc.reap_now() == 0
        assert svc.get_status()["lifecycle"]["reaped_total"] == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 8. concurrency limit behaviour
# ---------------------------------------------------------------------------


def test_limit_evicts_lru_idle_client(mock_registry):
    r1 = _make_repo(mock_registry, "r1")
    r2 = _make_repo(mock_registry, "r2")
    svc = _svc(max_servers=1)
    try:
        svc.get_diagnostics_sync(str(r1 / "x.py"))
        pid1 = _single_client(svc)["pid"]
        svc.get_diagnostics_sync(str(r2 / "x.py"))   # must evict r1, not fail
        st = svc.get_status()
        assert st["client_count"] == 1
        assert st["clients"][0]["workspace_root"] == str(r2)
        assert st["lifecycle"]["evicted_total"] == 1
        assert st["lifecycle"]["limit_refusals"] == 0
        assert _wait_until(lambda: not _pid_alive(pid1), timeout=5.0)
    finally:
        svc.shutdown()


def test_limit_refuses_spawn_when_all_clients_busy(mock_registry):
    r1 = _make_repo(mock_registry, "r1")
    r2 = _make_repo(mock_registry, "r2")
    svc = _svc(max_servers=1)
    try:
        svc.get_diagnostics_sync(str(r1 / "x.py"))
        key1 = ("pyright", str(r1))
        pid1 = _single_client(svc)["pid"]
        with svc._state_lock:
            svc._inflight[key1] = 1               # r1 is mid-request
        diags = svc.get_diagnostics_sync(str(r2 / "x.py"))
        assert diags == []                         # graceful fallback, no exception
        st = svc.get_status()
        assert st["client_count"] == 1
        assert st["clients"][0]["pid"] == pid1     # busy client untouched
        assert _pid_alive(pid1)
        assert st["lifecycle"]["limit_refusals"] == 1
        assert st["lifecycle"]["evicted_total"] == 0
        assert ("pyright", str(r2)) not in svc._broken   # refusal is not "broken"
    finally:
        with svc._state_lock:
            svc._inflight.pop(key1, None)
        svc.shutdown()


def test_per_id_limit_only_counts_same_server(mock_registry):
    r1 = _make_repo(mock_registry, "r1")
    r2 = _make_repo(mock_registry, "r2", ext=".ts")
    svc = _svc(max_servers=0, max_servers_per_id=1)
    try:
        svc.get_diagnostics_sync(str(r1 / "x.py"))
        svc.get_diagnostics_sync(str(r2 / "x.ts"))   # different server_id: allowed
        st = svc.get_status()
        assert st["client_count"] == 2
        assert {c["server_id"] for c in st["clients"]} == {"pyright", "typescript"}
        assert st["lifecycle"]["evicted_total"] == 0
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 9. workspace / root cleanup
# ---------------------------------------------------------------------------


def test_reaped_key_is_fully_removed_and_respawns(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.05)
    try:
        svc.get_diagnostics_sync(f)
        key = ("pyright", str(repo))
        pid1 = _single_client(svc)["pid"]
        time.sleep(0.1)
        assert svc.reap_now() == 1
        assert key not in svc._clients
        assert key not in svc._last_used
        assert key not in svc._inflight
        assert key not in svc._reaping
        assert key not in svc._broken
        assert svc.get_status()["client_count"] == 0
        # A later edit in the same workspace simply spawns a fresh server.
        svc.get_diagnostics_sync(f)
        c = _single_client(svc)
        assert c["pid"] != pid1 and _pid_alive(c["pid"])
        assert c["workspace_root"] == str(repo)
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# 10. gateway shutdown cleans all clients
# ---------------------------------------------------------------------------


def test_shutdown_stops_reaper_and_tears_down_all_clients(mock_registry):
    r1 = _make_repo(mock_registry, "r1")
    r2 = _make_repo(mock_registry, "r2", ext=".ts")
    svc = _svc(idle_timeout=3600.0, reap_interval=1.0, start_reaper=True)
    svc.get_diagnostics_sync(str(r1 / "x.py"))
    svc.get_diagnostics_sync(str(r2 / "x.ts"))
    st = svc.get_status()
    assert st["client_count"] == 2
    assert st["lifecycle"]["reaper_alive"] is True
    pids = [c["pid"] for c in st["clients"]]
    task = svc._reaper_task
    svc.shutdown()
    assert task.done()
    assert svc._reaper_task is None
    assert not svc._clients and not svc._inflight and not svc._last_used
    for pid in pids:
        assert _wait_until(lambda p=pid: not _pid_alive(p), timeout=5.0), f"pid {pid} survived shutdown"
    svc.shutdown()  # idempotent


def test_shutdown_service_singleton_path(monkeypatch, mock_registry):
    """The process-level ``shutdown_service()`` (atexit + gateway stop)
    must reach ``LSPService.shutdown`` and kill the server processes."""
    from agent import lsp as lsp_module

    repo = _make_repo(mock_registry, "r1")
    svc = _svc(reap_interval=1.0, start_reaper=True)
    monkeypatch.setattr(lsp_module, "_service", svc)
    monkeypatch.setattr(lsp_module, "_atexit_registered", True)
    svc.get_diagnostics_sync(str(repo / "x.py"))
    pid = _single_client(svc)["pid"]
    lsp_module.shutdown_service()
    assert lsp_module._service is None
    assert _wait_until(lambda: not _pid_alive(pid), timeout=5.0)


# ---------------------------------------------------------------------------
# 11. regression: an idle pyright cannot survive indefinitely
# ---------------------------------------------------------------------------


def test_regression_idle_pyright_does_not_survive_with_automatic_reaper(mock_registry):
    """Automatic reaper (not ``reap_now``): after one edit and no
    further use, the pyright process must be gone within
    ``idle_timeout + 2 * reap_interval``.  This is the 2026-09-08
    incident in miniature."""
    repo = _make_repo(mock_registry, "r1")
    f = str(repo / "x.py")
    svc = _svc(idle_timeout=0.3, reap_interval=1.0, start_reaper=True)
    try:
        svc.get_diagnostics_sync(f)
        c = _single_client(svc)
        assert c["server_id"] == "pyright" and _pid_alive(c["pid"])
        assert svc.get_status()["lifecycle"]["reaper_alive"] is True
        budget = 0.3 + 2 * 1.0 + 2.0
        assert _wait_until(lambda: svc.get_status()["client_count"] == 0, timeout=budget), (
            "idle pyright still registered after %.1fs" % budget
        )
        assert _wait_until(lambda: not _pid_alive(c["pid"]), timeout=5.0)
        assert svc.get_status()["lifecycle"]["reaped_total"] == 1
    finally:
        svc.shutdown()


# ---------------------------------------------------------------------------
# TAN-1039 regressions: the two HIGH findings from the retroactive review
# ---------------------------------------------------------------------------


def test_concurrent_same_key_spawn_at_cap_spawns_exactly_once(monkeypatch, tmp_path):
    """HIGH #1: at max_servers, two simultaneous requests for the same new
    key must not both evict-and-spawn.  The eviction victim is a 'stuck'
    server so `_make_room_for` genuinely suspends for the grace period,
    which is the window the race lived in.  Exactly one client object,
    one registered pid, and zero orphan mock processes for that root."""
    eventlog.reset_announce_caches()
    restore = _swap_server("pyright", "stuck")
    monkeypatch.chdir(str(tmp_path))
    r1 = _make_repo(tmp_path, "r1")
    r2 = _make_repo(tmp_path, "r2")
    f2 = str(r2 / "x.py")
    svc = _svc(max_servers=1, shutdown_grace=1.0)
    try:
        svc.get_diagnostics_sync(str(r1 / "x.py"))    # occupies the single slot
        assert svc.get_status()["client_count"] == 1

        async def _two():
            async def _one():
                async with svc._leased_client(f2) as c:
                    return c
            return await asyncio.gather(_one(), _one())

        c_a, c_b = svc._loop.run(_two(), timeout=30.0)
        assert c_a is not None and c_b is not None
        assert c_a is c_b, "two clients spawned for one key"
        st = svc.get_status()
        assert st["client_count"] == 1
        assert st["clients"][0]["workspace_root"] == str(r2)
        assert st["lifecycle"]["evicted_total"] == 1, "victim evicted more than once"
        assert _wait_until(lambda: len(_mock_pids_for_root(r2)) == 1, timeout=5.0), (
            f"orphan mock processes for r2: {_mock_pids_for_root(r2)}"
        )
        assert not svc._spawning and not svc._reaping
    finally:
        svc.shutdown()
        restore()


def test_stale_lease_on_crashed_client_does_not_uncount_replacement(monkeypatch, tmp_path):
    """HIGH #2: caller A holds a lease on client 1; the server dies; caller
    B spawns a replacement (lease count 1).  When A's lease is released it
    must not decrement the replacement's count, and the replacement must
    stay ineligible for reaping/eviction while B is in flight."""
    eventlog.reset_announce_caches()
    restore = _swap_server("pyright", "clean")
    monkeypatch.chdir(str(tmp_path))
    repo = _make_repo(tmp_path, "r1")
    f = str(repo / "x.py")
    key = ("pyright", str(repo))
    svc = _svc(idle_timeout=0.001)
    try:
        svc.get_diagnostics_sync(f)
        client1 = svc._clients[key]
        pid1 = client1.pid

        async def _scenario():
            # A takes a lease on client1 and keeps it.
            lease_a = svc._leased_client(f)
            got_a = await lease_a.__aenter__()
            assert got_a is client1
            assert svc._inflight[key] == 1
            # Server crashes under A.
            os.kill(pid1, 9)
            deadline = time.time() + 5
            while client1.is_running and time.time() < deadline:
                await asyncio.sleep(0.05)
            assert not client1.is_running
            # B arrives: dead-client branch spawns a replacement with its own lease.
            lease_b = svc._leased_client(f)
            client2 = await lease_b.__aenter__()
            assert client2 is not None and client2 is not client1
            assert svc._clients[key] is client2
            assert svc._inflight[key] == 1, "replacement lease not counted"
            # A releases its stale lease: replacement must remain counted.
            await lease_a.__aexit__(None, None, None)
            assert svc._inflight.get(key) == 1, "stale lease decremented the replacement"
            with svc._state_lock:
                assert svc._select_idle(time.time() + 10, min_idle=0.0) == [], (
                    "busy replacement selectable as a victim"
                )
            assert await svc._reap_idle() == 0
            assert svc._clients[key] is client2 and client2.is_running
            await lease_b.__aexit__(None, None, None)
            assert key not in svc._inflight
            return client2.pid

        pid2 = svc._loop.run(_scenario(), timeout=30.0)
        assert pid2 != pid1 and _pid_alive(pid2)
    finally:
        svc.shutdown()
        restore()


def test_real_lease_blocks_reap_during_request(monkeypatch, tmp_path):
    """A request that is genuinely in flight (slow diagnostics, no private
    state injected) survives a reap pass with idle_timeout ~0."""
    import threading

    eventlog.reset_announce_caches()
    restore = _swap_server("pyright", "clean")
    monkeypatch.chdir(str(tmp_path))
    repo = _make_repo(tmp_path, "r1")
    slow = repo / "slow_file.py"
    slow.write_text("x = 1\n")
    svc = _svc(idle_timeout=0.001, wait_timeout=4.0)
    result = {}

    def _caller():
        result["diags"] = svc.get_diagnostics_sync(str(slow), delta=False)

    try:
        t = threading.Thread(target=_caller, daemon=True)
        t.start()
        assert _wait_until(lambda: svc.get_status()["client_count"] == 1, timeout=5.0)
        pid = _single_client(svc)["pid"]
        time.sleep(0.3)                                   # inside the 1.0s slow window
        assert _single_client(svc)["inflight"] == 1
        assert svc.reap_now() == 0, "reaped a client with a real in-flight request"
        assert _pid_alive(pid)
        t.join(timeout=10)
        assert not t.is_alive() and "diags" in result
        assert _single_client(svc)["pid"] == pid
        time.sleep(0.05)
        assert svc.reap_now() == 1                        # idle now: reaped
    finally:
        svc.shutdown()
        restore()


def test_shutdown_grace_zero_escalates_immediately(stuck_registry):
    repo = _make_repo(stuck_registry, "r1")
    svc = _svc(idle_timeout=0.05, shutdown_grace=0)
    try:
        svc.get_diagnostics_sync(str(repo / "x.py"))
        pid = _single_client(svc)["pid"]
        time.sleep(0.1)
        t0 = time.time()
        assert svc.reap_now() == 1
        assert time.time() - t0 < 2.0, "grace=0 did not escalate immediately"
        assert _wait_until(lambda: not _pid_alive(pid), timeout=5.0)
        assert svc.get_status()["lifecycle"]["forced_total"] == 1
    finally:
        svc.shutdown()


def test_reaper_survives_failing_pass(mock_registry, monkeypatch):
    repo = _make_repo(mock_registry, "r1")
    svc = _svc(idle_timeout=0.05, reap_interval=1.0, start_reaper=True)
    try:
        svc.get_diagnostics_sync(str(repo / "x.py"))
        calls = {"n": 0}
        real = svc._shutdown_client

        async def _boom(client):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic teardown failure")
            return await real(client)

        monkeypatch.setattr(svc, "_shutdown_client", _boom)
        # First pass raises inside the victim coroutine; the key is released
        # from _reaping and the client is already unregistered, so the
        # process is collected by shutdown() below. The reaper stays alive.
        assert _wait_until(lambda: calls["n"] >= 1, timeout=5.0)
        time.sleep(0.2)
        lc = svc.get_status()["lifecycle"]
        assert lc["reaper_alive"] is True
        assert not svc._reaping
    finally:
        svc.shutdown()


def test_per_id_cap_evicts_only_same_server(mock_registry):
    r1 = _make_repo(mock_registry, "r1")
    r2 = _make_repo(mock_registry, "r2", ext=".ts")
    r3 = _make_repo(mock_registry, "r3")
    svc = _svc(max_servers=0, max_servers_per_id=1)
    try:
        svc.get_diagnostics_sync(str(r1 / "x.py"))
        svc.get_diagnostics_sync(str(r2 / "x.ts"))
        ts_pid = next(c["pid"] for c in svc.get_status()["clients"] if c["server_id"] == "typescript")
        svc.get_diagnostics_sync(str(r3 / "x.py"))       # per-id cap on pyright: evict r1, not typescript
        st = svc.get_status()
        roots = {c["server_id"]: c["workspace_root"] for c in st["clients"]}
        assert roots == {"pyright": str(r3), "typescript": str(r2)}
        assert st["lifecycle"]["evicted_total"] == 1
        assert _pid_alive(ts_pid)
    finally:
        svc.shutdown()


def test_defaults_are_bounded():
    """Guard against someone restoring 'servers live forever'."""
    assert 0 < DEFAULT_IDLE_TIMEOUT <= 24 * 3600
    assert 0 < DEFAULT_REAP_INTERVAL <= DEFAULT_IDLE_TIMEOUT
    assert DEFAULT_MAX_SERVERS > 0


def test_reaper_starts_by_default_and_config_is_honoured(monkeypatch):
    cfg = {
        "lsp": {
            "enabled": True,
            "idle_timeout": 123,
            "reap_interval": 7,
            "shutdown_grace": 3,
            "max_servers": 2,
            "max_servers_per_id": "not-a-number",   # falls back to default, no crash
        }
    }
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    svc = LSPService.create_from_config()
    try:
        lc = svc.get_status()["lifecycle"]
        assert lc["idle_timeout"] == 123.0
        assert lc["reap_interval"] == 7.0
        assert lc["shutdown_grace"] == 3.0
        assert lc["max_servers"] == 2
        assert lc["max_servers_per_id"] == 3
        assert _wait_until(lambda: svc.get_status()["lifecycle"]["reaper_alive"], timeout=2.0)
    finally:
        svc.shutdown()


def test_idle_timeout_zero_disables_reaper(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    svc = _svc(idle_timeout=0, start_reaper=True)
    try:
        svc.get_diagnostics_sync(str(repo / "x.py"))
        time.sleep(0.1)
        assert svc.get_status()["lifecycle"]["reaper_alive"] is False
        assert svc.reap_now() == 0
        assert svc.get_status()["client_count"] == 1
    finally:
        svc.shutdown()


def test_status_exposes_process_evidence(mock_registry):
    repo = _make_repo(mock_registry, "r1")
    svc = _svc()
    try:
        svc.get_diagnostics_sync(str(repo / "x.py"))
        c = _single_client(svc)
        assert c["pid"] and c["created_at"] and c["last_used_at"]
        assert c["age_seconds"] >= 0 and c["idle_seconds"] >= 0
        assert c["inflight"] == 0
        if os.path.exists("/proc/self/status"):
            assert c["rss_kb"] and c["rss_kb"] > 0
            assert c["swap_kb"] is not None
    finally:
        svc.shutdown()
