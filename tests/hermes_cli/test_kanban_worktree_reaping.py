"""Tests for reap_stale_worktrees.

The function deletes directories, so the tests are weighted toward what it must
REFUSE to do. Every gate gets a case proving the worktree survives, because the
failure mode that matters is not "failed to reclaim disk" -- it is "reclaimed
disk and took someone's work with it".
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=True)


def make_repo_with_worktree(tmp_path, name="wt", *, push=True):
    """A repo with a remote and a linked worktree, mirroring the real shape."""
    remote = tmp_path / f"{name}-remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo = tmp_path / f"{name}-repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    git(repo, "remote", "add", "origin", str(remote))
    (repo / "seed.txt").write_text("seed\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "seed")
    if push:
        git(repo, "push", "-q", "origin", "main")
    wt = tmp_path / f"{name}-worktree"
    # A distinct branch per worktree: git refuses to check out the same
    # branch in two worktrees, and real kanban worktrees are per-task anyway.
    git(repo, "worktree", "add", "-q", "-b", f"task/{name}", str(wt), "main")
    if push:
        git(repo, "fetch", "-q", "origin")
    return repo, wt


def add_task(conn, task_id, path, *, status="done", age_days=30, kind="worktree"):
    completed = int(time.time()) - age_days * 86400
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, completed_at, "
        "workspace_kind, workspace_path, branch_name) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (task_id, "t", status, completed, completed, kind, str(path), f"task/{task_id}"),
    )
    conn.commit()


def test_reaps_a_long_finished_clean_fully_pushed_worktree(kanban_home, tmp_path):
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_reap", wt)

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert [r["task_id"] for r in report["reaped"]] == ["t_reap"], report
    assert not wt.exists(), "the worktree directory should be gone"


def test_dry_run_is_the_default_and_removes_nothing(kanban_home, tmp_path):
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_dry", wt)

    report = kb.reap_stale_worktrees(conn)

    assert report["dry_run"] is True
    assert report["reaped"] and report["reaped"][0]["removed"] is False
    assert wt.exists(), "dry run must not delete anything"


def test_uncommitted_work_is_never_reaped(kanban_home, tmp_path):
    """The 2026-08-02 lesson: work that only exists in a working tree is the
    work most easily destroyed, so it is exactly what must survive a sweep."""
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    (wt / "in-progress.txt").write_text("not committed anywhere\n")
    add_task(conn, "t_dirty", wt)

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert wt.exists()
    reason = report["skipped"][0]["reason"]
    assert "uncommitted work" in reason
    assert "preserve" in reason, "the skip must point at the preserve tooling"


def test_commits_on_no_remote_are_never_reaped(kanban_home, tmp_path):
    conn = kb.connect()
    repo, wt = make_repo_with_worktree(tmp_path)
    (wt / "local-only.txt").write_text("committed but never pushed\n")
    git(wt, "add", "-A")
    git(wt, "commit", "-qm", "local only")
    add_task(conn, "t_unpushed", wt)

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert wt.exists()
    assert "on no remote" in report["skipped"][0]["reason"]


def test_recently_completed_worktrees_are_left_alone(kanban_home, tmp_path):
    """Retention-by-default is the point; only retention-forever is the bug."""
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_fresh", wt, age_days=1)

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert report["skipped"] == [], "too-recent tasks are filtered in SQL, not skipped"
    assert wt.exists()


def test_unfinished_tasks_are_left_alone(kanban_home, tmp_path):
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_open", wt, status="in_progress")

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert wt.exists()


def test_scratch_workspaces_are_none_of_this_functions_business(kanban_home, tmp_path):
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_scratch", wt, kind="scratch")  # scratch is removed on completion already

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == [] and report["skipped"] == []
    assert wt.exists()


def test_an_unreadable_worktree_is_refused_rather_than_guessed_at(kanban_home, tmp_path):
    """A git command that cannot answer must block the reap. Treating silence
    as 'nothing there' is how a sweep deletes something it never inspected.

    The genuine case is a LINKED worktree whose gitdir has gone away: it still
    looks like a worktree (.git is a file), so it passes the path-type gate, but
    no git command against it can succeed. ~15 such dirs exist on the live fleet.
    """
    conn = kb.connect()
    broken = tmp_path / "dangling-worktree"
    broken.mkdir()
    (broken / ".git").write_text(f"gitdir: {tmp_path}/gone/.git/worktrees/dangling\n")
    (broken / "work.txt").write_text("someone's files\n")
    add_task(conn, "t_broken", broken)

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert broken.exists()
    assert "refusing to guess" in report["skipped"][0]["reason"]


def test_a_plain_directory_is_skipped_not_deleted(kanban_home, tmp_path):
    """workspace_kind='dir' is in scope, but a plain directory is NOT removable
    here: that needs shutil.rmtree plus a containment guard (#28818). On the
    live board 88 of 102 eligible rows are exactly this, so the skip must be
    explicit rather than incidental."""
    conn = kb.connect()
    plain = tmp_path / "plain-dir"
    plain.mkdir()
    (plain / "data.txt").write_text("not a git worktree at all\n")
    add_task(conn, "t_plain", plain, kind="dir")

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert plain.exists(), "a plain directory must never be deleted by this function"
    assert "not a linked git worktree" in report["skipped"][0]["reason"]


def test_dir_kind_worktrees_are_now_in_scope(kanban_home, tmp_path):
    """The bug this fixes: the live board records its t_<hex> git worktrees as
    kind='dir', so filtering on kind='worktree' matched 0 rows of 302 and the
    reaper was silently inert."""
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_dirkind", wt, kind="dir")

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert [r["task_id"] for r in report["reaped"]] == ["t_dirkind"], report
    assert not wt.exists()


def test_a_non_terminal_child_blocks_the_parents_reap(kanban_home, tmp_path):
    conn = kb.connect()
    _repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_parent", wt)
    add_task(conn, "t_child", tmp_path / "child", status="in_progress", kind="scratch")
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?,?)",
                 ("t_parent", "t_child"))
    conn.commit()

    report = kb.reap_stale_worktrees(conn, dry_run=False)

    assert report["reaped"] == []
    assert wt.exists()
    assert "child" in report["skipped"][0]["reason"]


def test_the_branch_survives_a_reap(kanban_home, tmp_path):
    """Only the directory is reclaimed. The branch is a ref that costs nothing
    and is what keeps the work recoverable -- deleting both would turn a space
    reclaim into data loss."""
    conn = kb.connect()
    repo, wt = make_repo_with_worktree(tmp_path)
    add_task(conn, "t_branch", wt)

    kb.reap_stale_worktrees(conn, dry_run=False)

    assert not wt.exists()
    assert git(repo, "rev-parse", "--verify", "task/wt").returncode == 0
