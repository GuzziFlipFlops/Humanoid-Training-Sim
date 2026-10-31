"""CPU-only saved-evidence review; never starts Isaac or modifies a run."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def yaw(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))


def review_trace(folder):
    p = folder / "gait_diagnostics_50hz.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        times = z["time"]
        heading = yaw(z["root_state"][..., 3:7])
        target = z["heading_target"]
        command = z["action_command"]
        error = np.arctan2(np.sin(target-heading), np.cos(target-heading))
        moving = command[..., 0] > .1
        first_alive = np.cumsum(z["done"], axis=0) == 0
        mask = moving & first_alive & (times[:, None] >= 2.)
        angular_z = z["root_state"][..., 12]
        desired = command[..., 2]
        desired_observed = z["observed_command"]
        # Response sign is a diagnostic, not a causal actuator test. Both command
        # and yaw are coupled through the closed-loop policy and heading error.
        meaningful = mask & (np.abs(desired) > .03)
        heading_change = np.arctan2(np.sin(heading[1:]-heading[:-1]),
                                   np.cos(heading[1:]-heading[:-1]))
        measured_rate = heading_change / np.diff(times)[:, None]
        angular_consistency = np.abs(measured_rate-angular_z[1:])
        cases = []
        for speed in np.unique(z["commands"]):
            ids = z["commands"] == speed
            case_mask = mask[:, ids]
            if not case_mask.any():
                continue
            values = lambda a: a[:, ids][case_mask]
            vcmd, vresp = values(desired), values(angular_z)
            active = np.abs(vcmd) > .03
            cases.append(dict(speed=float(speed), robot_ids=np.flatnonzero(ids).tolist(),
                sampled_live_frames=int(case_mask.sum()),
                heading_abs_mean_deg=float(np.abs(values(error)).mean()*180/np.pi),
                yaw_command_mean=float(vcmd.mean()), yaw_response_mean=float(vresp.mean()),
                yaw_rate_tracking_rms=float(np.sqrt(np.mean((vcmd-vresp)**2))),
                nontrivial_yaw_frames=int(active.sum()),
                yaw_correct_sign_fraction=float(np.mean(vcmd[active]*vresp[active]>0.)) if active.any() else None,
                yaw_command_bound_fraction=float(np.mean(np.abs(vcmd)>=.19999))))
        return dict(path=str(p), scope="First 16 saved robot traces; full population gates are reported separately.",
            observed_action_command_max_difference=float(np.max(np.abs(command-desired_observed))),
            heading_target_max_variation=float(np.ptp(target, axis=0).max()),
            world_yaw_rate_vs_finite_heading_p95=float(np.quantile(angular_consistency[first_alive[1:]], .95)) if first_alive[1:].any() else None,
            cases=cases)


def main():
    active = read(TRAINING / "ACTIVE_RUN.json")
    run = Path(active["run"])
    assert run.resolve().parent == (TRAINING / "runs").resolve()
    state = read(run / "state.json")
    records, gates, train = [], [], []
    for folder in sorted(p for p in run.iterdir() if p.is_dir() and p.name[:3].isdigit()):
        rows = []
        if (folder / "progress.jsonl").exists():
            for line in (folder / "progress.jsonl").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Active append may be incomplete; don't alter the live file.
                    continue
        if rows:
            rows = [r for r in rows if r.get("collect_seconds", 0)>0]
            if rows:
                train.extend(rows)
                timings = [r["collect_seconds"]+r["learn_seconds"] for r in rows]
                record = dict(folder=folder.name, updates=len(rows), environments=rows[0]["num_envs"],
                    active_steps_s=statistics.mean(r["num_envs"]*24/t for r,t in zip(rows,timings)),
                    collection_fraction=statistics.mean(r["collect_seconds"]/t for r,t in zip(rows,timings)),
                    minimum_host_free_mb=min(r["gpu"]["ram_available_mb"] for r in rows),
                    minimum_vram_free_mb=min(r["gpu"]["free_mb"] for r in rows),
                    max_pre_update_kl=max(r["ppo"]["pre_update_kl"] for r in rows),
                    post_update_kl_mean=statistics.mean(r["ppo"]["post_update_kl"] for r in rows),
                    last_return=rows[-1]["ppo"]["return_mean"],
                    native_memory_samples=[dict(iteration=r["iteration"], time=r["time"],
                        host_available=r["gpu"]["ram_available_mb"], private=r["gpu"]["process_private_mb"],
                        rss=r["gpu"]["process_rss_mb"], torch_allocated=r["gpu"]["torch_allocated_mb"])
                        for r in rows],
                    last_behavior=rows[-1].get("behavior"), last_joints=rows[-1].get("joints"))
                records.append(record)
        if "gate_" in folder.name and (folder / "result.json").exists():
            result = read(folder / "result.json")
            gates.append(dict(folder=folder.name, passed=result.get("passed"),
                cases=[dict(passed=c["passed"], failures=c["failures"], statistics=c["statistics"])
                       for c in result["cases"]],
                trace=review_trace(folder)))
    top=[]
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            mem=p.memory_info()
            if mem.rss>500*2**20:
                top.append(dict(pid=p.pid,name=p.name(),rss_mb=round(mem.rss/2**20),
                    private_mb=round(getattr(mem,"private",0)/2**20)))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    evidence = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),
        state={k:state.get(k) for k in ["status","stage","stage_speed","operation_path","checkpoint",
            "total_training_iterations","environment_count","promotions","vram_ramp","resource_recycles"]},
        operations=records,gates=gates,
        top_processes=sorted(top,key=lambda p:p["private_mb"],reverse=True),
        scope="Saved artifacts plus read-only current process memory; no simulator, optimizer or run changes.")
    (HERE/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(dict(recorded_utc=evidence["recorded_utc"],status=state["status"],stage=state["stage"],
        completed_main_updates=state.get("total_training_iterations"),
        latest_gates=[dict(folder=g["folder"],passed=g["passed"],cases=[dict(speed=c["statistics"]["command_m_s"],
            passed=c["passed"],failed=c["failures"],survival=c["statistics"]["survival"],
            body_speed=c["statistics"]["forward_speed_m_s"],
            fixed_forward=c["statistics"].get("reset_frame_forward_speed_m_s"),
            fixed_lateral=c["statistics"].get("reset_frame_lateral_speed_m_s"),
            final_heading_deg=c["statistics"].get("final_heading_abs_mean_deg"),
            heading_outside=c["statistics"].get("heading_outside_fraction"),
            steps=[c["statistics"]["steps_left"],c["statistics"]["steps_right"]]) for c in g["cases"]])
            for g in gates],top_processes=evidence["top_processes"],output=str(HERE/"evidence.json")),indent=2))


if __name__ == "__main__":
    main()
