"""CPU-only capture of deployed training evidence; never starts Isaac."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(read(HERE / "ACTIVE_RUN.json")["run"])
assert run.name == "locomotion_20261004_r33_qualified_walking_retention"
state = read(run / "state.json")
blocks = []
for folder in sorted(run.glob("*_run_150_train_*")):
    rows = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()] if (folder / "progress.jsonl").exists() else []
    proof = read(folder / "full_load_proof.json") if (folder / "full_load_proof.json").exists() else None
    certificate = read(folder / "qualified_prior_certificate.json")
    teacher = Path(certificate["checkpoint"])
    assert sha(teacher) == certificate["checkpoint_sha256"]
    assert certificate["maximum_command_m_s"] == 1.0
    if rows:
        assert all(x["ppo"]["pre_update_kl"] == 0 for x in rows)
        assert all(x["behavior"]["action_command_difference"] == 0 for x in rows)
        assert all(x["locomotion_retention"]["batches"] == 20 for x in rows)
        assert all(x["locomotion_retention"]["command_range"] == [0.1, 1.0] for x in rows)
    if proof:
        assert all(proof[k] for k in ("actor_equal", "critic_equal", "optimizer_equal", "exploration_equal"))
    gpu = [x["gpu"] for x in rows]
    total_active = sum(x["collect_seconds"] + x["learn_seconds"] for x in rows)
    rollout_steps = read(folder / "agent.json")["num_steps_per_env"]
    sample_count = sum(x["num_envs"] * rollout_steps for x in rows)
    result = read(folder / "result.json") if (folder / "result.json").exists() else None
    attempt = next((a for a in state["attempts"] if Path(a["folder"]).resolve() == folder.resolve()), None)
    worker_samples = result["iterations"] * result["num_envs"] * rollout_steps if result else None
    blocks.append({
        "folder": str(folder), "observed_updates": len(rows),
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "certificate": certificate, "full_load_proof": proof,
        "max_pre_update_kl": max((x["ppo"]["pre_update_kl"] for x in rows), default=None),
        "max_command_mismatch": max((x["behavior"]["action_command_difference"] for x in rows), default=None),
        "last_locomotion_retention": rows[-1]["locomotion_retention"] if rows else None,
        "min_free_gpu_mb": min((x["free_mb"] for x in gpu), default=None),
        "min_host_available_mb": min((x["ram_available_mb"] for x in gpu), default=None),
        "private_growth_mb": gpu[-1]["process_private_mb"] - gpu[0]["process_private_mb"] if gpu else None,
        "torch_allocated_growth_mb": gpu[-1]["torch_allocated_mb"] - gpu[0]["torch_allocated_mb"] if gpu else None,
        "active_samples_per_second": sample_count / total_active if total_active else None,
        "worker_effective_samples_per_second": worker_samples / attempt["seconds"] if worker_samples and attempt and attempt["completed"] else None,
        "checkpoint_sha256": sha(Path(result["checkpoint"])) if result else None,
        "completed_result": result,
    })
gates = []
for folder in sorted(run.glob("*_gate_*")):
    if not (folder / "result.json").exists():
        continue
    result = read(folder / "result.json")
    config = read(folder / "run_config.json")
    gates.append({
        "folder": str(folder), "passed": result["passed"], "run_config": config,
        "checkpoint_sha256": sha(Path(config["arguments"]["checkpoint"])),
        "cases": [{
            "command_m_s": c["statistics"]["command_m_s"], "passed": c["passed"],
            "forward_speed_m_s": c["statistics"]["reset_frame_forward_speed_m_s"],
            "geometric_flight_fraction": c["statistics"].get("geometric_flight_fraction"),
            "statistics": c["statistics"], "failed_checks": [k for k, v in c["checks"].items() if not v],
        } for c in result["cases"]],
    })
data = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
    "status": state["status"], "stage": state["stage"],
    "completed_main_updates": state["total_training_iterations"],
    "operation": state.get("operation_path"), "blocks": blocks, "gates": gates,
    "scope": "Progress is training execution evidence; only matching complete independent gates qualify behavior.",
}
(OUT / "new_main_evidence.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps({
    "recorded_utc": data["recorded_utc"], "operation": data["operation"],
    "completed_main_updates": data["completed_main_updates"],
    "blocks": [{k: v for k, v in b.items() if k not in ("certificate", "completed_result")} for b in blocks],
    "gates": [{"folder": Path(g["folder"]).name, "passed": g["passed"], "cases": [{k:v for k,v in c.items() if k != "statistics"} for c in g["cases"]]} for g in gates],
}, indent=2))
