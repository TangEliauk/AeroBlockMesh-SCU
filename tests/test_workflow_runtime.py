"""Local workflow boundaries: frozen data, isolated outputs and real artifacts.

All writable fixtures are created and removed inside this implementation copy.
No test trains a policy, contacts a remote, or invokes an external CFD solver.
"""
from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from smartcfd.core.types import Failure
from smartcfd.pipeline import config as run_config
from smartcfd.pipeline import workflow
from smartcfd.pipeline.data import data_version, load_domains, load_samples, repository_root


@pytest.fixture
def local_workspace():
    root = repository_root().resolve()
    base = root / ".runtime" / "workflow_test_fixtures"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case_", dir=base) as temporary:
        path = Path(temporary).resolve()
        assert path.is_relative_to(root) and path != root
        yield path
        # Verify the recursive-cleanup target before TemporaryDirectory exits.
        assert path.resolve().is_relative_to(root) and path.resolve() != root


@pytest.fixture
def fixture_repository(local_workspace, monkeypatch):
    root = local_workspace / "repository"
    root.mkdir()
    monkeypatch.setattr(run_config, "repository_root", lambda: root)
    monkeypatch.setattr(run_config, "data_version", lambda: {"version": "fixture_only"})
    return root


@pytest.fixture(scope="module")
def one_train_domain():
    # An actual training geometry; the held-out geometries are never meshed here.
    return load_domains("train", limit=1)[0]


def arguments(output, **overrides):
    values = {"output": str(output), "split": "train", "limit": 1, "seed": 0,
              "device": "cpu", "checkpoint": None, "method": "rule"}
    values.update(overrides)
    return Namespace(**values)


def test_sharp_geometry_counts_and_split_disjointness_are_frozen():
    samples = {split: load_samples(split) for split in ("train", "val", "test")}
    ids = {split: {s["sample_id"] for s in rows} for split, rows in samples.items()}
    assert {split: len(rows) for split, rows in samples.items()} == {
        "train": 780, "val": 103, "test": 90,
    }
    assert all(len(ids[split]) == len(rows) for split, rows in samples.items())
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_new_run_never_overwrites_an_existing_run(fixture_repository):
    destination = fixture_repository / "runs" / "first"
    output = run_config.new_run("test", {"version": 1}, destination)
    marker = output / "keep.bin"
    marker.write_bytes(b"preserve existing experiment")
    receipt = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_config.new_run("test", {"version": 2}, destination)
    assert marker.read_bytes() == b"preserve existing experiment"
    assert (output / "run.json").read_bytes() == receipt


def test_prepare_writes_only_run_manifests_and_preserves_frozen_data(local_workspace):
    before = data_version()
    run, result = workflow.prepare({}, arguments(local_workspace / "prepared_lists"))
    assert result["sharp_geometry_counts"] == {"train": 780, "val": 103, "test": 90}
    for split, count in result["sharp_geometry_counts"].items():
        ids = json.loads((run / f"{split}.json").read_text(encoding="utf-8"))
        assert len(ids) == len(set(ids)) == count
    assert data_version() == before
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["status"] == "completed"


@pytest.mark.parametrize("absolute", [False, True])
def test_new_run_rejects_escape_before_any_directory_creation(
        fixture_repository, local_workspace, absolute):
    # The requested escape from this fake repository still lies inside the real
    # implementation copy, even if a broken confinement check allows the write.
    escaped = local_workspace / ("outside_absolute" if absolute else "outside_relative")
    output = escaped if absolute else Path("../outside_relative")
    with pytest.raises(ValueError):
        run_config.new_run("test", {}, output)
    assert not escaped.exists()


def test_read_config_rejects_typo_in_top_level_field(local_workspace):
    path = local_workspace / "invalid.yaml"
    path.write_text('{"envv": {"max_steps": 4}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        run_config.read_config(path)


@pytest.mark.parametrize("bc_only", [False, True])
def test_test_split_is_refused_before_allocating_a_run_or_trainer(monkeypatch, bc_only):
    def must_not_run(*args, **kwargs):
        raise AssertionError("Test split reached an executable training operation")
    monkeypatch.setattr(workflow, "new_run", must_not_run)
    monkeypatch.setattr(workflow, "create_trainer", must_not_run)
    monkeypatch.setattr(workflow, "load_domains", must_not_run)
    with pytest.raises(ValueError, match="Test geometries"):
        workflow.train({}, Namespace(split="test"), bc_only=bc_only)


@pytest.mark.parametrize("operation", [workflow.generate, workflow.evaluate])
def test_test_mesh_and_evaluation_require_explicit_final_snapshot(monkeypatch, operation):
    def must_not_run(*args, **kwargs):
        raise AssertionError("Unfrozen test access reached runtime or data loading")
    monkeypatch.setattr(workflow, "new_run", must_not_run)
    monkeypatch.setattr(workflow, "load_domains", must_not_run)
    with pytest.raises(ValueError, match="final-test"):
        operation({}, arguments(None, split="test", final_test=False))


def test_teacher_failure_is_preserved_without_fake_mesh_artifacts(
        local_workspace, one_train_domain, monkeypatch):
    from smartcfd.baselines import rule_partition
    failure = Failure("fixture_teacher_failure", "specific failed rule",
                      {"mesh_reason": "negative_jacobian", "variant": 1})
    monkeypatch.setattr(rule_partition, "solve", lambda *a, **k: failure)
    monkeypatch.setattr(workflow, "load_domains", lambda *a, **k: [one_train_domain])
    run, result = workflow.generate({}, arguments(local_workspace / "failed_generation"))
    assert result["success_rate"] == 0
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert record["success"] is False and record["reason"] == failure.reason
    assert record["detail"] == failure.detail
    assert record["diagnostics"] == failure.diagnostics
    assert not (run / one_train_domain.sample_id).exists()
    receipt = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert receipt["result"]["records"][0]["success"] is False


def test_final_test_checkpoint_cannot_resume_training(local_workspace, monkeypatch):
    import hashlib
    root = local_workspace
    checkpoint = root / 'opaque.pt'
    checkpoint.write_bytes(b'guard must reject this before deserializing')
    manifest = root / 'experiments/runs/final/final_test_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}), encoding='utf-8')
    monkeypatch.setattr(workflow, 'repository_root', lambda: root)
    args = Namespace(device='cpu', checkpoint=str(checkpoint), task='train')
    with pytest.raises(ValueError, match='frozen for final test'):
        workflow.create_trainer({}, args)


def test_all_failed_demonstrations_still_write_review_receipt_and_return_failures(
        local_workspace, one_train_domain, monkeypatch):
    from smartcfd.baselines import rule_partition
    failure = Failure("fixture_no_valid_mesh", "both requested edge counts conflict",
                      {"q10_scaled_jacobian": -0.2})
    monkeypatch.setattr(rule_partition, "solve", lambda *a, **k: failure)
    demo_path = local_workspace / "failed_demonstrations.jsonl"
    review = local_workspace / "new_review_directory"
    transitions, report = workflow.collect_demonstrations(
        {}, [one_train_domain], 2, 0, demo_path, review)
    assert transitions == [] and report["successful"] == 0
    assert report["requested"] == 2 and len(report["failures"]) == 2
    assert all(row["reason"] == failure.reason for row in report["failures"])
    assert all(row["diagnostics"] == failure.diagnostics for row in report["failures"])
    assert demo_path.read_text(encoding="utf-8") == ""
    assert json.loads((review / "review_manifest.json").read_text(encoding="utf-8")) == []
    assert (review / "index.html").is_file()


@pytest.mark.parametrize("script", [
    "prepare_data.py", "generate_mesh.py", "pretrain_imitation.py", "train_rl.py",
    "evaluate.py", "run_cfd.py", "visualize.py",
])
def test_cli_help_exits_without_training_or_configuration_execution(script):
    root = repository_root()
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    process = subprocess.run([sys.executable, "-B", str(root / "scripts" / script), "--help"],
                             cwd=root, env=environment, capture_output=True,
                             text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stderr
    assert "usage:" in process.stdout.lower()
    assert "--output" in process.stdout and "--split" in process.stdout


def test_mesh_cli_creates_real_npz_json_and_su2_consistently(local_workspace):
    root = repository_root()
    destination = local_workspace / "real_mesh_command"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    process = subprocess.run([
        sys.executable, "-B", str(root / "scripts" / "generate_mesh.py"),
        "--output", str(destination), "--split", "train", "--limit", "1", "--seed", "0",
    ], cwd=root, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    console = json.loads(process.stdout)
    assert Path(console["run_directory"]).resolve() == destination.resolve()
    receipt = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    records = receipt["result"]["records"]
    assert len(records) == 1 and records[0]["success"]
    sample_dir = destination / records[0]["sample_id"]
    manifest = json.loads((sample_dir / "mesh.json").read_text(encoding="utf-8"))
    assert manifest["quality"]["valid"]
    assert manifest["quality"]["q10_scaled_jacobian"] > 0
    with np.load(sample_dir / "mesh.npz", allow_pickle=False) as arrays:
        assert len(arrays.files) == records[0]["blocks"] == len(manifest["blocks"])
        cells = 0
        for name in arrays.files:
            data = arrays[name]
            assert data.ndim == 3 and data.shape[-1] == 2 and np.isfinite(data).all()
            cells += (data.shape[0] - 1) * (data.shape[1] - 1)
    su2 = (sample_dir / "mesh.su2").read_text(encoding="ascii")
    assert "NDIME= 2" in su2 and f"NELEM= {cells}\n" in su2
    assert "MARKER_TAG= wall" in su2 and "MARKER_TAG= farfield" in su2
    graph = json.loads((sample_dir / "graph.json").read_text(encoding="utf-8"))
    assert len(graph["faces"]) == records[0]["blocks"] == 8
