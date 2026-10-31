"""Refresh compact closing receipts without replacing exact-pair handoffs."""
import json
from datetime import datetime, timezone
from pathlib import Path
import psutil

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

inventory = read(OUT / "inventory.log")
assert inventory["differing_manifest_hashes"] == 0
assert inventory["read_warnings"] == []
auto = read(OUT / "automation_verification.json")
assert all(auto[k] for k in ["same_thread", "five_hour_cadence", "notification_preference_preserved", "durable_scope_endpoint_prefix_unchanged", "prompt_matches_args"])
assert auto["status"] == "ACTIVE"
active = read(TRAIN / "ACTIVE_RUN.json")
state = read(Path(active["run"]) / "state.json")
pause = read(OUT / "diagnostic_pause.json")
sup = psutil.Process(39000)
assert sup.create_time() == pause["supervisor_create_time"]
assert sup.status() != psutil.STATUS_STOPPED and state["status"] == "running"
workers = [p for p in psutil.process_iter(["cmdline"]) if any(Path(s).name == "worker_unitree.py" for s in p.info["cmdline"] or [])]
leaves = {p.pid for p in workers} - {p.ppid() for p in workers}
assert len(leaves) == 1
assert all(Path(p.cmdline()[p.cmdline().index("--output") + 1]).resolve() == Path(state["operation_path"]).resolve() for p in workers)
assert not any(any(Path(s).name in ["timing_probe.py", "run_timing_probe.py", "run_diagnosis.py"] for s in p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))
write(OUT / "inventory_verification.json", inventory)
closure = read(OUT / "closeout_verification.json")
closure.update(recorded_utc=datetime.now(timezone.utc).isoformat(), inventory_checks=inventory,
    actual_worker=next(iter(leaves)), venv_worker_launcher_pids=[p.pid for p in workers if p.pid not in leaves],
    operation=state["operation_path"], native_automation_active_same_thread_5h=True,
    reviewed_gates_and_upper="Exact431 candidate with matching432/433 and newer_upper timelines reviewed. Gates do not certify the current newer learning actor.",
    automation_verification=str(OUT / "automation_verification.json"))
write(OUT / "closeout_verification.json", closure)
n = read(TRAIN / "automation/NEXT_CHECK.json")
n.update(automation_verified_active=True, automation_verification=str(OUT / "automation_verification.json"),
    native_automation_verification=str(OUT / "automation_verification.json"), latest_schedule_verification=str(OUT / "automation_verification.json"),
    closeout_verification=str(OUT / "closeout_verification.json"), history_inventory=str(OUT / "inventory_verification.json"),
    closing_inventory=str(OUT / "inventory_verification.json"), closing_liveness=str(OUT / "closeout_verification.json"))
write(TRAIN / "automation/NEXT_CHECK.json", n)
with (OUT / "REPORT.md").open("a", encoding="utf-8") as h:
    h.write(f"\nFinal metadata receipt {closure['recorded_utc']}: native automation reverified ACTIVE on the same thread at five hours with the newest431 handoff and preserved durable scope/notifications/endpoint. CPU inventory now contains {inventory['physical_operations']} operations across40 physical runs, zero differing hashes/read warnings, and eight documented historical missing manifests. Compact closing receipt replaces the mistakenly copied artifact list with this integrity summary. Same supervisor39000 has one actual worker and no diagnostic process; current learning operation is `{state['operation_path']}`. Reviewed431 qualification and posture remain separate from this newer actor.\n")
print(json.dumps({"automation": auto, "inventory": inventory, "supervisor": 39000, "actual_worker": closure["actual_worker"], "operation": state["operation_path"], "production_change": False}, indent=2))
