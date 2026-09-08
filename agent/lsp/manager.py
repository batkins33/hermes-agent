"""Service-level orchestration for LSP clients.

The :class:`LSPService` is the bridge between the synchronous
file_operations layer and the async :class:`agent.lsp.client.LSPClient`.

Design choices:

- A **single asyncio event loop** runs in a background thread.  All
  client work happens on that loop.  Synchronous callers from
  ``tools/file_operations.py`` use :meth:`get_diagnostics_sync` to
  open + wait + drain in one blocking call.

- One client per ``(server_id, workspace_root)`` key.  Lazy spawn:
  the first request for a key spawns the client; subsequent requests
  re-use it.  Clients are shared by every session/caller in the
  process that touches the same key — ownership is per workspace, not
  per session.

- A **broken-set** records ``(server_id, workspace_root)`` pairs that
  failed to spawn or initialize.  These are never retried for the
  life of the service.  Mirrors OpenCode's design.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

- **Bounded lifecycle.**  Every use of a client is a *lease*
  (``_leased_client``) that bumps an in-flight counter for the
  duration of the call and refreshes ``last_used``.  A reaper task on
  the same loop wakes every ``reap_interval`` seconds and shuts down
  clients that have had no lease for ``idle_timeout`` seconds.  A
  client with an active lease is never reaped.  Spawning is capped by
  ``max_servers`` (total) and ``max_servers_per_id`` (per language
  server); when a cap is hit the least-recently-used *idle* client is
  evicted first, and if every client is busy the spawn is refused
  (the caller falls back to the in-process syntax check) rather than
  killing something that is mid-request.

  Why this exists: on 2026-09-08 a ``pyright-langserver`` child of the
  Hermes gateway survived ~98 hours with no workspace files open, no
  CPU activity, 2.1 GB RSS and 2.3 GB of swap, because ``idle_timeout``
  and ``_last_used`` were recorded but never consumed.  The gateway is
  a long-lived daemon, so "servers live for the life of the process"
  meant "servers live forever".

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
)
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

# Lifecycle defaults.  All overridable via ``lsp.*`` in config.yaml —
# see ``hermes_cli/config.py``.  Chosen so an interactive session that
# pauses for a coffee keeps its index, while a daemon that stops
# editing a workspace releases the server within the hour.
DEFAULT_IDLE_TIMEOUT = 1800.0      # seconds without a lease before a client is reaped
DEFAULT_REAP_INTERVAL = 60.0       # seconds between reaper passes
DEFAULT_SHUTDOWN_GRACE = 10.0      # seconds to wait for graceful shutdown before SIGKILL
DEFAULT_MAX_SERVERS = 8            # total live clients; 0 = unlimited
DEFAULT_MAX_SERVERS_PER_ID = 3     # live clients per server_id; 0 = unlimited
MIN_REAP_INTERVAL = 1.0            # floor so a typo can't spin the loop

_Key = Tuple[str, str]


class _BackgroundLoop:
    """A daemon thread that owns one asyncio event loop.

    Provides :meth:`run` for synchronous callers — submits a coroutine
    to the loop and blocks until it finishes (or a timeout fires).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hermes-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


class LSPService:
    """The process-wide LSP service.

    Created once via :meth:`create_from_config`; the
    :func:`agent.lsp.get_service` accessor manages the singleton.
    Most callers should use that accessor rather than constructing
    :class:`LSPService` directly.
    """

    # ------------------------------------------------------------------
    # construction + factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: Optional[Dict[str, List[str]]] = None,
        env_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        init_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disabled_servers: Optional[List[str]] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        reap_interval: float = DEFAULT_REAP_INTERVAL,
        shutdown_grace: float = DEFAULT_SHUTDOWN_GRACE,
        max_servers: int = DEFAULT_MAX_SERVERS,
        max_servers_per_id: int = DEFAULT_MAX_SERVERS_PER_ID,
        start_reaper: bool = True,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])

        # Lifecycle policy.  ``idle_timeout <= 0`` disables idle reaping
        # (clients live until shutdown); caps of 0 mean unlimited.
        self._idle_timeout = float(idle_timeout)
        self._reap_interval = max(MIN_REAP_INTERVAL, float(reap_interval))
        self._shutdown_grace = max(0.0, float(shutdown_grace))
        self._max_servers = max(0, int(max_servers))
        self._max_servers_per_id = max(0, int(max_servers_per_id))

        self._loop = _BackgroundLoop()

        # Per-(server_id, workspace_root) state.  Everything below is
        # guarded by ``_state_lock`` because the sync callers, the loop
        # thread and the reaper all touch it.
        self._clients: Dict[_Key, LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[_Key, asyncio.Future] = {}
        self._last_used: Dict[_Key, float] = {}
        self._inflight: Dict[_Key, int] = {}
        self._reaping: set = set()   # keys currently being torn down
        self._state_lock = threading.Lock()

        # Lifecycle counters (monotonic for the life of the service).
        self._reaped_total = 0
        self._evicted_total = 0
        self._graceful_failures = 0
        self._forced_total = 0
        self._limit_refusals = 0
        self._reaper_errors = 0
        self._last_reap_at: Optional[float] = None
        self._reaper_task: Optional[asyncio.Task] = None

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        if self._enabled:
            self._loop.start()
            if start_reaper:
                self._start_reaper()

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        servers_cfg = lsp_cfg.get("servers") or {}
        disabled = []
        binary_overrides: Dict[str, List[str]] = {}
        env_overrides: Dict[str, Dict[str, str]] = {}
        init_overrides: Dict[str, Dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                cmd = sub.get("command")
                if isinstance(cmd, list) and cmd:
                    binary_overrides[name] = cmd
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {k: str(v) for k, v in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        def _num(key: str, default: float, cast=float):
            raw = lsp_cfg.get(key, default)
            try:
                return cast(raw)
            except (TypeError, ValueError):
                logger.warning("lsp.%s=%r is not a number; using %s", key, raw, default)
                return cast(default)

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            idle_timeout=_num("idle_timeout", DEFAULT_IDLE_TIMEOUT),
            reap_interval=_num("reap_interval", DEFAULT_REAP_INTERVAL),
            shutdown_grace=_num("shutdown_grace", DEFAULT_SHUTDOWN_GRACE),
            max_servers=_num("max_servers", DEFAULT_MAX_SERVERS, int),
            max_servers_per_id=_num("max_servers_per_id", DEFAULT_MAX_SERVERS_PER_ID, int),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (file or cwd inside a git worktree),
        on whether any registered server matches the extension, and
        on whether the (server_id, workspace_root) pair is in the
        broken-set from a previous spawn failure.

        Files in already-broken pairs return False so the file_operations
        layer skips the LSP path entirely — no spawn attempts, no
        timeout cost — until the service is restarted (``hermes lsp
        restart``) or the process exits.
        """
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _get_or_spawn would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if (srv.server_id, per_server_root) in self._broken:
            return False
        return True

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer timeouts (e.g. server hangs during initialize) mark the
        (server_id, workspace_root) pair as broken so subsequent edits
        skip it instantly instead of re-paying the timeout cost.
        """
        if not self.enabled_for(file_path):
            return
        try:
            diags = self._loop.run(self._snapshot_async(file_path), timeout=8.0)
            self._delta_baseline[os.path.abspath(file_path)] = diags or []
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            self._delta_baseline[os.path.abspath(file_path)] = []

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default), the result is filtered against
        any baseline previously captured via :meth:`snapshot_baseline`.
        Diagnostics present in the baseline are removed so the caller
        only sees errors introduced by the current edit.

        When ``line_shift`` is provided, baseline diagnostics are
        remapped through it before the set-difference.  This handles
        the case where the edit deleted or inserted lines, causing
        pre-existing diagnostics below the edit point to surface at
        different line numbers in the post-edit snapshot — without
        the shift, they'd all look "introduced by this edit".  Pass
        a callable built by
        :func:`agent.lsp.range_shift.build_line_shift` (pre_text,
        post_text).  Omit when pre/post content isn't available;
        the unshifted comparison still catches diagnostics that
        didn't move.

        Returns an empty list when LSP is disabled, when no workspace
        can be detected, when no server matches, or when the server
        can't be spawned.  Never raises.
        """
        if not self.enabled_for(file_path):
            return []

        # Resolve server_id eagerly so we can emit structured logs even
        # when the request errors out below.
        srv = find_server_for_file(file_path)
        server_id = srv.server_id if srv else "?"

        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t) or []
        except asyncio.TimeoutError as e:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []

        abs_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(abs_path) or []
            if baseline:
                if line_shift is not None:
                    # Remap baseline diagnostics into post-edit
                    # coordinates so shifted-but-otherwise-identical
                    # entries hash equal under _diag_key.  Entries
                    # that mapped into a deleted region drop out
                    # silently — they no longer apply.
                    from agent.lsp.range_shift import shift_baseline
                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(d) for d in baseline}
                diags = [d for d in diags if _diag_key(d) not in seen]
            # Roll baseline forward — next call returns deltas relative
            # to the just-emitted state, mirroring claude-code's
            # diagnosticTracking.
            try:
                fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[abs_path] = fresh

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_get_or_spawn`` task was still
        holding open.  Without this, every subsequent write would re-enter
        the spawn path and re-pay the full ``snapshot_baseline``
        timeout (8s) until the binary is fixed.

        Also kills any orphan client process that survived the cancelled
        future, and emits a single eventlog WARNING so the user knows
        which server gave up.

        ``exc`` is whatever exception the outer wrapper caught — used
        only for logging, never re-raised.
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            return
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        key = (srv.server_id, per_server_root)
        already_broken = key in self._broken
        self._broken.add(key)

        # Kill any client we managed to spawn before the timeout.  The
        # cancelled future never reached the broken-set add inside
        # ``_get_or_spawn`` so the client may still be hanging in
        # ``_clients`` with a half-initialized state.
        client = self._forget_client(key)
        if client is not None:
            try:
                # Fire-and-forget shutdown — give it a second to cleanup,
                # but don't block.  We're already on a slow path.
                self._loop.run(client.shutdown(), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

    def shutdown(self) -> None:
        """Tear down all clients and stop the background loop.

        Order matters: the reaper is cancelled first so it cannot race
        the final sweep, then every client is shut down (gracefully,
        then forced), then the loop stops.  Idempotent.
        """
        if not self._enabled:
            return
        try:
            self._loop.run(self._stop_reaper(), timeout=5.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP reaper stop error: %s", e)
        try:
            self._loop.run(self._shutdown_async(), timeout=self._shutdown_grace + 10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
        self._loop.stop()
        clear_cache()

    def reap_now(self) -> int:
        """Run one reaper pass synchronously.  Returns clients reaped.

        Exposed for ``hermes lsp status``-style tooling and tests; the
        background reaper calls the same coroutine.
        """
        if not self._enabled or self._loop.loop is None:
            return 0
        try:
            return int(self._loop.run(self._reap_idle(), timeout=self._shutdown_grace + 15.0) or 0)
        except Exception as e:  # noqa: BLE001
            # Tooling entry point: never raise out of a reap request.
            self._reaper_errors += 1
            logger.warning("LSP reap_now failed: %s", e)
            return 0

    # ------------------------------------------------------------------
    # async internals — leases
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _leased_client(self, file_path: str) -> AsyncIterator[Optional[LSPClient]]:
        """Yield the client for ``file_path`` with an in-flight lease held.

        The lease is taken *under the state lock in the same critical
        section that hands out the client*, so the reaper can never
        observe "idle" for a client a caller is about to use.  The
        lease is released — and ``last_used`` refreshed — on exit,
        including on exception.
        """
        client = await self._get_or_spawn(file_path)
        if client is None:
            yield None
            return
        key = (client.server_id, client.workspace_root)
        try:
            yield client
        finally:
            with self._state_lock:
                # Bind the release to the client object, not just the key:
                # if this client died mid-request and a replacement was
                # spawned under the same key, the replacement's lease count
                # belongs to *its* callers and must not be decremented here
                # (TAN-1039 HIGH #2).
                if self._clients.get(key) is client:
                    n = self._inflight.get(key, 1) - 1
                    if n > 0:
                        self._inflight[key] = n
                    else:
                        self._inflight.pop(key, None)
                    self._last_used[key] = time.time()

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        async with self._leased_client(file_path) as client:
            if client is None:
                return []
            try:
                version = await client.open_file(file_path, language_id=language_id_for(file_path))
                await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
            except Exception as e:  # noqa: BLE001
                logger.debug("snapshot open/wait failed: %s", e)
                return []
            return list(client.diagnostics_for(file_path))

    async def _open_and_wait_async(self, file_path: str) -> List[Dict[str, Any]]:
        async with self._leased_client(file_path) as client:
            if client is None:
                return []
            try:
                version = await client.open_file(file_path, language_id=language_id_for(file_path))
                await client.save_file(file_path)
                await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
            except Exception as e:  # noqa: BLE001
                logger.debug("open/wait failed for %s: %s", file_path, e)
                return []
            return list(client.diagnostics_for(file_path))

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            client = self._clients.get((srv.server_id, ws))
        if client is None:
            return []
        return list(client.diagnostics_for(file_path))

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        """Return a running client for ``file_path`` with one lease taken.

        Callers MUST release the lease (see :meth:`_leased_client`).
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        per_server_root = srv.resolve_root(file_path, ws_root)
        if per_server_root is None:
            eventlog.log_disabled(
                srv.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None  # exclude marker hit, server gated off

        key = (srv.server_id, per_server_root)
        if key in self._broken:
            return None
        # Lookup, dead-client replacement, and spawn registration happen
        # in ONE critical section.  The spawn future is registered here,
        # before any await (including the cap check below), so a second
        # same-key request can only ever find and await this future —
        # never start a parallel spawn (TAN-1039 HIGH #1).
        loop = asyncio.get_running_loop()
        spawn_future: asyncio.Future = loop.create_future()
        with self._state_lock:
            client = self._clients.get(key)
            if client is not None and client.is_running:
                # Lease taken in the same critical section as lookup —
                # the reaper can't slip in between.
                self._inflight[key] = self._inflight.get(key, 0) + 1
                self._last_used[key] = time.time()
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            if client is not None:
                # Dead client (server crashed).  Drop it so the spawn
                # below replaces it instead of returning a corpse.  Any
                # lease still held on the corpse releases against the
                # object, not the key (see _leased_client).
                self._clients.pop(key, None)
                self._inflight.pop(key, None)
            spawning = self._spawning.get(key)
            if spawning is None:
                self._spawning[key] = spawn_future
        if spawning is not None:
            try:
                spawned = await spawning
            except Exception:  # noqa: BLE001
                return None
            if spawned is None:
                return None
            with self._state_lock:
                if self._clients.get(key) is not spawned or not spawned.is_running:
                    return None
                self._inflight[key] = self._inflight.get(key, 0) + 1
                self._last_used[key] = time.time()
            return spawned

        # We own the spawn for this key.  Everything below runs under the
        # registered future; the finally pops it.
        try:
            # Enforce concurrency caps BEFORE paying for a spawn.  Evicts
            # the least-recently-used idle client if that frees a slot;
            # refuses (never kills a busy client) otherwise.  In-flight
            # spawns (other keys) count toward the caps.
            if not await self._make_room_for(key):
                spawn_future.set_result(None)
                return None
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            )
            try:
                await client.start()
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            with self._state_lock:
                self._clients[key] = client
                self._inflight[key] = self._inflight.get(key, 0) + 1
                self._last_used[key] = time.time()
            eventlog.log_active(srv.server_id, per_server_root)
            spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                self._spawning.pop(key, None)

    # ------------------------------------------------------------------
    # async internals — lifecycle (reaper + caps)
    # ------------------------------------------------------------------

    def _start_reaper(self) -> None:
        """Schedule the reaper task on the background loop (idempotent)."""
        if self._idle_timeout <= 0:
            logger.debug("LSP idle reaper disabled (idle_timeout=%s)", self._idle_timeout)
            return
        if self._loop.loop is None:
            return

        async def _create() -> None:
            if self._reaper_task is None or self._reaper_task.done():
                self._reaper_task = asyncio.get_running_loop().create_task(
                    self._reaper_loop(), name="hermes-lsp-reaper"
                )

        # Synchronous so ``get_status()['lifecycle']['reaper_alive']`` is
        # truthful the moment the constructor returns.
        try:
            self._loop.run(_create(), timeout=5.0)
        except Exception as e:  # noqa: BLE001
            self._reaper_errors += 1
            logger.warning("LSP idle reaper failed to start: %s", e)

    async def _stop_reaper(self) -> None:
        task = self._reaper_task
        self._reaper_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _reaper_loop(self) -> None:
        logger.debug(
            "LSP idle reaper started (idle_timeout=%.0fs, interval=%.0fs, max=%d/%d per id)",
            self._idle_timeout, self._reap_interval, self._max_servers, self._max_servers_per_id,
        )
        while True:
            await asyncio.sleep(self._reap_interval)
            try:
                await self._reap_idle()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # The reaper must survive a bad cycle; a dead reaper is
                # exactly the failure mode this module exists to prevent.
                self._reaper_errors += 1
                eventlog.log_reaper_error(e)

    def _select_idle(self, now: float, *, min_idle: float) -> List[_Key]:
        """Return keys with no lease held and idle >= ``min_idle`` seconds.

        Must be called with ``_state_lock`` held.  Oldest-idle first.
        """
        out = []
        for key, client in self._clients.items():
            if key in self._reaping or key in self._spawning:
                continue
            if self._inflight.get(key, 0) > 0:
                continue
            idle = now - self._last_used.get(key, now)
            if idle >= min_idle:
                out.append((idle, key))
        out.sort(reverse=True)
        return [k for _, k in out]

    def _forget_client(self, key: _Key) -> Optional[LSPClient]:
        """Atomically remove ``key`` from live state.  Returns the client
        (if any) so the caller can shut it down outside the lock."""
        with self._state_lock:
            client = self._clients.pop(key, None)
            self._last_used.pop(key, None)
            self._inflight.pop(key, None)
            return client

    async def _reap_idle(self) -> int:
        """One reaper pass.  Returns the number of clients reclaimed."""
        if self._idle_timeout <= 0:
            return 0
        now = time.time()
        with self._state_lock:
            victims = self._select_idle(now, min_idle=self._idle_timeout)
            # Claim atomically: a key in ``_reaping`` is invisible to a
            # concurrent pass, and it is popped from ``_clients`` here
            # so no new lease can be taken on it.
            claimed = []
            for key in victims:
                client = self._clients.pop(key, None)
                if client is None:
                    continue
                self._reaping.add(key)
                idle = now - self._last_used.pop(key, now)
                self._inflight.pop(key, None)
                claimed.append((key, client, idle))
        self._last_reap_at = now
        if not claimed:
            return 0

        async def _one(key: _Key, client: LSPClient, idle: float) -> bool:
            try:
                forced = await self._shutdown_client(client)
                self._reaped_total += 1
                eventlog.log_reaped(key[0], key[1], idle, forced=forced, reason="idle")
                return True
            finally:
                with self._state_lock:
                    self._reaping.discard(key)

        # Victims are torn down concurrently so a pass is bounded by one
        # grace period, not N of them (TAN-1039 WARN).
        results = await asyncio.gather(
            *(_one(k, c, i) for k, c, i in claimed), return_exceptions=True
        )
        return sum(1 for r in results if r is True)

    async def _shutdown_client(self, client: LSPClient) -> bool:
        """Graceful shutdown with a bounded wait, then SIGKILL.

        Returns True if termination had to be forced (either the
        graceful path exceeded ``shutdown_grace`` or the client itself
        had to escalate to SIGKILL).
        """
        forced = False
        # Run the graceful path as its own task and *wait* on it rather
        # than cancelling it: cancelling would still block on the
        # client's internal SIGTERM grace inside its ``finally``, which
        # made the bound soft.  On timeout we SIGKILL directly — the
        # graceful task then unblocks on its own (the pipes close and
        # ``proc.wait()`` returns) and is collected below.
        task = asyncio.ensure_future(client.shutdown())
        # shutdown_grace == 0 means "escalate to SIGKILL immediately";
        # it is never an unbounded wait (TAN-1039 WARN).
        if self._shutdown_grace > 0:
            done, _pending = await asyncio.wait({task}, timeout=self._shutdown_grace)
        else:
            done = set()
        if task not in done:
            self._graceful_failures += 1
            forced = True
            with contextlib.suppress(Exception):
                await client.kill_now()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except Exception:  # noqa: BLE001
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        else:
            exc = task.exception()
            if exc is not None:
                self._graceful_failures += 1
                forced = True
                logger.debug("LSP graceful shutdown of %s raised: %s", client.server_id, exc)
                with contextlib.suppress(Exception):
                    await client.kill_now()
        if client.last_shutdown_forced:
            forced = True
        if forced:
            self._forced_total += 1
        return forced

    async def _make_room_for(self, key: _Key) -> bool:
        """Apply ``max_servers`` / ``max_servers_per_id`` before a spawn.

        Evicts least-recently-used *idle* clients (no lease held) until
        the caps allow ``key``.  Returns False — and logs once per key —
        when the caps are hit and nothing idle can be evicted.  A busy
        client is never terminated to make room.
        """
        while True:
            with self._state_lock:
                # Live clients plus spawns already in flight for OTHER keys
                # (ours is registered too; don't count ourselves).
                pending = [k for k in self._spawning if k != key]
                total = len(self._clients) + len(pending)
                per_id = sum(1 for k in self._clients if k[0] == key[0]) + sum(
                    1 for k in pending if k[0] == key[0]
                )
                over_total = self._max_servers and total >= self._max_servers
                over_id = self._max_servers_per_id and per_id >= self._max_servers_per_id
                if not (over_total or over_id):
                    return True
                candidates = self._select_idle(time.time(), min_idle=0.0)
                if over_id and not over_total:
                    candidates = [k for k in candidates if k[0] == key[0]]
                if not candidates:
                    self._limit_refusals += 1
                    break
                # LRU = largest idle age = first after the sort in _select_idle.
                victim = candidates[0]
                client = self._clients.pop(victim, None)
                self._reaping.add(victim)
                idle = time.time() - self._last_used.pop(victim, time.time())
                self._inflight.pop(victim, None)
            try:
                if client is not None:
                    forced = await self._shutdown_client(client)
                    self._evicted_total += 1
                    eventlog.log_reaped(victim[0], victim[1], idle, forced=forced, reason="limit")
            finally:
                with self._state_lock:
                    self._reaping.discard(victim)
        eventlog.log_limit_reached(
            key[0], key[1], total=total, max_total=self._max_servers,
            per_id=per_id, max_per_id=self._max_servers_per_id,
        )
        return False

    async def _shutdown_async(self) -> None:
        with self._state_lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._broken.clear()
            self._last_used.clear()
            self._inflight.clear()
            # Anything a concurrent reaper claimed is already on its way
            # out; it is not in ``_clients`` any more.
        results = await asyncio.gather(
            *(self._shutdown_client(c) for c in clients),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.debug("LSP client shutdown error: %s", r)

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command.

        ``clients`` carries per-process evidence (pid, age, idle age,
        RSS/swap where ``/proc`` is available) so a future "why is the
        gateway holding 4 GB" question can be answered with one
        ``hermes lsp status --json``.
        """
        now = time.time()
        # Snapshot under the lock; sample /proc outside it so status never
        # holds the hot-path lock during file I/O.
        with self._state_lock:
            snapshot = [
                (k, c, self._last_used.get(k), self._inflight.get(k, 0))
                for k, c in self._clients.items()
            ]
            broken = list(self._broken)
            reaper_alive = self._reaper_task is not None and not self._reaper_task.done()
        clients = []
        for k, c, last, inflight in snapshot:
            mem = c.memory_usage()
            clients.append(
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                    "pid": c.pid,
                    "created_at": c.created_at,
                    "age_seconds": round(now - c.created_at, 1) if c.created_at else None,
                    "last_used_at": last,
                    "idle_seconds": round(now - last, 1) if last else None,
                    "inflight": inflight,
                    "rss_kb": mem.get("rss_kb"),
                    "swap_kb": mem.get("swap_kb"),
                }
            )
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "clients": clients,
            "client_count": len(clients),
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
            "lifecycle": {
                "idle_timeout": self._idle_timeout,
                "reap_interval": self._reap_interval,
                "shutdown_grace": self._shutdown_grace,
                "max_servers": self._max_servers,
                "max_servers_per_id": self._max_servers_per_id,
                "reaper_alive": reaper_alive,
                "last_reap_at": self._last_reap_at,
                "reaped_total": self._reaped_total,
                "evicted_total": self._evicted_total,
                "graceful_failures": self._graceful_failures,
                "forced_total": self._forced_total,
                "limit_refusals": self._limit_refusals,
                "reaper_errors": self._reaper_errors,
            },
        }


def _diag_key(d: Dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering.

    Includes the diagnostic's position range — when used together
    with :func:`agent.lsp.range_shift.shift_baseline`, the baseline
    is line-shifted into post-edit coordinates BEFORE this key is
    computed, so identical-but-shifted diagnostics hash equal.  Two
    genuinely distinct diagnostics at different lines (e.g. the same
    error class introduced at a second site) hash differently and
    are surfaced as new.

    Mirrors :func:`agent.lsp.client._diagnostic_key`; intentionally
    identical so the two layers agree on diagnostic identity.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
