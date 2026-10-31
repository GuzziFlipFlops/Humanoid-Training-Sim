"""CPU-only documentation inventory. Never imports Isaac/Torch or changes a run.

Run from any directory with the workspace Python. Only generated files in this
history directory are written; the human narratives are deliberately untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

HISTORY = Path(__file__).resolve().parent
TRAINING = HISTORY.parent
WORKSPACE = TRAINING.parent.parent
OLD = WORKSPACE / "humanoid_project"
CAD = WORKSPACE / "outputs/shin_cad_20260912"
WARNINGS: list[str] = []
HASH_CACHE: dict[str, str] = {}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"items": value}
    except (ValueError, OSError) as exc:
        WARNINGS.append(f"{path}: {exc}")
        return {"inventory_read_error": str(exc)}


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    key = str(path)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def utc(timestamp) -> str | None:
    if not isinstance(timestamp, (int, float)):
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def file_record(path: Path, hashed=False) -> dict:
    item = {"path": path.as_posix(), "bytes": path.stat().st_size,
            "modified_utc": utc(path.stat().st_mtime)}
    if hashed:
        item["sha256"] = digest(path)
    return item


def link(path: Path | str, label: str | None = None) -> str:
    path = Path(path)
    if not path.exists():
        return f"`{path.as_posix()}` (target not retained at this path)"
    try:
        target = Path(os.path.relpath(path, HISTORY)).as_posix()
    except ValueError:  # Another Windows drive.
        target = path.as_posix()
    return f"[{label or path.name}](<{target}>)"


def flat(text) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def run_number(path: Path) -> int:
    m = re.search(r"_r(\d+)(?:_|$)", path.name)
    return int(m.group(1)) if m else 1


def frozen_path(run: Path, key: str) -> Path:
    root = run / "source"
    if key in ("supervisor.py", "supervisor_unitree.py", "vram_ramp.py") and (run / key).exists():
        return run / key
    if key.startswith("physical_v29/"):
        preferred = root / key
    elif key.startswith("physical_assets/"):
        preferred = root / "physical_v29/assets" / key.split("/", 1)[1]
    else:
        preferred = root / "physical_v29/training" / key
    if preferred.exists():
        return preferred
    alternatives = [root / key, root / "physical_v29" / key]
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    matches = list(root.rglob(Path(key).name)) if root.exists() else []
    return matches[0] if len(matches) == 1 else preferred


def operations(run: Path) -> list[dict]:
    answer = []
    for folder in sorted((p for p in run.iterdir() if p.is_dir()
                          and p.name not in ("source", "source_snapshot", "__pycache__")),
                         key=lambda p: (0, int(p.name.split('_', 1)[0]), p.name)
                         if p.name.split('_', 1)[0].isdigit() else (1, 0, p.name)):
        files = sorted(p for p in folder.iterdir() if p.is_file())
        cfg = read_json(folder / "run_config.json")
        result = read_json(folder / "result.json")
        failure = read_json(folder / "failure.json")
        args = cfg.get("arguments", {})
        # No result is a real missing/unfinished record, not an implicit success.
        disposition = ("failed" if failure else "result_recorded" if result
                       else "no_result_recorded")
        answer.append({"folder": folder.as_posix(), "name": folder.name,
                       "disposition": disposition, "configuration": cfg,
                       "result": result, "failure": failure,
                       "input_checkpoint": args.get("checkpoint"),
                       "files": [file_record(p) for p in files]})
    return answer


def inspect_physical(run: Path) -> dict:
    state = read_json(run / "state.json")
    config = read_json(run / "chain_config.json")
    ops = operations(run)
    manifest = config.get("sources_sha256", {})
    verification = []
    for key, expected in manifest.items():
        path = frozen_path(run, key)
        actual = digest(path)
        verification.append({"key": key, "path": path.as_posix(),
                             "expected_sha256": expected, "actual_sha256": actual,
                             "matches": actual == expected})
    interfaces = []
    for op in ops:
        cfg = op["configuration"]
        interface = {k: cfg[k] for k in ("contract", "reward_version", "gate_version",
                    "reference_runtime", "mass_kg", "urdf_sha256", "actuator_sha256",
                    "motion_sha256", "physics_dt", "control_dt", "self_collision") if k in cfg}
        if interface and interface not in interfaces:
            interfaces.append(interface)
    checkpoint = state.get("checkpoint")
    item = {"revision": run_number(run), "run": run.as_posix(),
            "state_file": (run / "state.json").as_posix(),
            "state": state, "chain_config": config,
            "started_utc": utc(state.get("started_unix")),
            "updated_utc": utc(state.get("updated_unix")),
            "interfaces_observed": interfaces, "operations": ops,
            "frozen_source_verification": verification,
            "last_gate": read_json(run / "latest_gate.json"),
            "root_evidence_files": [file_record(p) for p in sorted(run.iterdir())
                                    if p.is_file() and p.suffix.lower() in (".json", ".md")],
            "state_checkpoint_sha256": digest(Path(checkpoint)) if checkpoint else None}
    return item


def inspect_legacy(run: Path) -> dict:
    state = read_json(run / "state.json")
    checkpoints = sorted(run.glob("*.pt"))
    folders = sorted(p for p in run.iterdir() if p.is_dir())
    blocks = []
    for folder in folders:
        if folder.name in ("source_snapshot", "source", "__pycache__"):
            continue
        files = [p for p in folder.iterdir() if p.is_file()]
        blocks.append({"folder": folder.as_posix(),
                       "result": read_json(folder / "result.json"),
                       "evaluation": read_json(folder / "evaluation.json"),
                       "failure": read_json(folder / "failure.json"),
                       "files": [file_record(p) for p in sorted(files)]})
    root_json = sorted(p for p in run.glob("*.json"))
    gates = {p.name: read_json(p) for p in root_json if "gate" in p.name}
    if checkpoints:
        checkpoints.sort(key=lambda p: int(re.search(r"(\d+)\.pt$", p.name).group(1))
                         if re.search(r"(\d+)\.pt$", p.name) else -1)
    return {"run": run.as_posix(), "state": state, "gates": gates,
            "root_files": [file_record(p) for p in sorted(run.iterdir()) if p.is_file()],
            "blocks": blocks, "root_checkpoint_count": len(checkpoints),
            "first_root_checkpoint": checkpoints[0].as_posix() if checkpoints else None,
            "last_root_checkpoint": checkpoints[-1].as_posix() if checkpoints else None,
            "qualification_scope": "Historical saved evidence, not a live-process or modern-gate certification"}


def evidence_tree(root: Path, exclude=()) -> list[dict]:
    if not root.exists():
        return []
    entries = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in exclude:
            continue
        # Inventory evidence and authoring scripts; do not traverse dependency or
        # converted-mesh caches, images, frozen runtimes or large model files.
        files = []
        for current, dirs, names in os.walk(child):
            dirs[:] = sorted(d for d in dirs if d not in (
                "node_modules", "__pycache__", ".git", "source", "source_snapshot",
                "cad_mesh_env", "meshes", "usd", "configuration"))
            for name in sorted(names):
                path = Path(current) / name
                if path.suffix.lower() in (".md", ".json", ".html", ".py", ".ps1", ".log"):
                    files.append(file_record(path))
        entries.append({"folder": child.as_posix(), "files": files})
    return entries


def write_physical_details(records: list[dict]) -> None:
    out = ["# Physical training run evidence", "",
           "Generated by `record_inventory.py`. Human rationale is in [the main history](../ITERATION_HISTORY.md).",
           "Saved status is historical; consult ACTIVE_RUN.json and actual processes for liveness.", "",
           "Counts are the root state's recorded main updates. Smoke/benchmark and unfinished-periodic",
           "work is catalogued separately. Inherited/rolled-back updates must not be summed as one actor's age.", ""]
    for row in records:
        n = row["revision"]
        run = Path(row["run"])
        state = row["state"]
        config = row["chain_config"]
        out.extend([f"## R{n:02d} — {run.name}", "",
                    f"{link(run / 'state.json', 'State')} · {link(run / 'chain_config.json', 'Configuration')} · {link(run / 'source', 'Frozen source')}", "",
                    f"Recorded status: `{state.get('status', 'not recorded')}`; stage: `{state.get('stage', 'not recorded')}`; "
                    f"main updates: **{state.get('total_training_iterations', 0)}**; "
                    f"inherited-count field: **{state.get('prior_training_iterations', 'not recorded')}**.", "",
                    f"Recorded start UTC: {row['started_utc'] or 'not recorded'}; last state UTC: {row['updated_utc'] or 'not recorded'}.", ""])
        lineage = {k: config.get(k) for k in ("resume_run", "migrate_run", "reward_migrate_run",
                   "initial_actor_checkpoint") if config.get(k)}
        for key, value in lineage.items():
            out.append(f"- `{key}`: {link(value)}")
        checkpoint = state.get("checkpoint")
        if checkpoint:
            out.extend(["", f"State checkpoint: {link(checkpoint)}.", "",
                        f"SHA256 at inventory: `{row['state_checkpoint_sha256'] or 'file missing'}`."])
        if row["interfaces_observed"]:
            out.extend(["", "Recorded worker interfaces (missing fields are not reconstructed):", ""])
            for interface in row["interfaces_observed"]:
                out.append("- " + "; ".join(f"`{k}={v}`" for k, v in interface.items()))
        promos = state.get("promotions", [])
        out.extend(["", "Recorded promotions: " + (", ".join(str(p.get("stage")) for p in promos) or "none") + "."])
        if n == 25:
            out.extend(["", "**R25 promotions use the old direction-permissive verifier. They do not qualify straight walking under the later fixed-frame gates.**"])
        for promotion in promos:
            if promotion.get("checkpoint"):
                out.append(f"- `{promotion.get('stage')}` checkpoint: {link(promotion['checkpoint'])}")
            if promotion.get("evidence"):
                out.append(f"- Matching evidence: {link(promotion['evidence'])}")
        verified = row["frozen_source_verification"]
        good = sum(p["matches"] for p in verified)
        out.extend(["", f"Frozen source manifest: **{good}/{len(verified)}** recorded hashes match at inventory."])
        for issue in (p for p in verified if not p["matches"]):
            out.append(f"- Missing or differing artifact: `{issue['key']}`; original manifest is unchanged.")
        changes = row.get("manifest_difference", {})
        if changes:
            out.extend(["", f"Manifest comparison against R{changes['compared_to']:02d} ({changes['basis']}):"])
            for key in ("changed", "added", "removed"):
                out.append(f"- {key}: " + (", ".join(f"`{p}`" for p in changes[key]) or "none"))
        error = state.get("error")
        if error:
            out.extend(["", "Root-state failure (preserved verbatim in inventory JSON):", "",
                        "```text", str(error).splitlines()[-1], "```"])
        out.extend(["", "### Saved worker operations", "",
                    "Execution completion is separate from behavioral qualification; a completed evaluation can fail.", "",
                    "| Operation | Execution evidence | Behavioral result / iterations | Configuration |",
                    "| --- | --- | --- | --- |"])
        for op in row["operations"]:
            folder = Path(op["folder"])
            result, failure = op["result"], op["failure"]
            if failure:
                status = link(folder / "failure.json", "failure")
            elif result:
                status = link(folder / "result.json", "result")
            else:
                status = "No result recorded"
            behavior = (f"Gate passed={result['passed']}" if "passed" in result else
                        f"{result.get('iterations', '?')} updates; completed={result.get('completed', '?')}; recycled={result.get('recycled', False)}"
                        if result else flat(failure.get("error", failure.get("reason", ""))))
            cfgpath = folder / "run_config.json"
            cfg = link(cfgpath, "config") if cfgpath.exists() else "Not recorded"
            out.append(f"| {link(folder, op['name'])} | {status} | {flat(behavior)} | {cfg} |")
        out.append("")
    (HISTORY / "PHYSICAL_RUNS.md").write_text("\n".join(out), encoding="utf-8")


def write_legacy_details(records: list[dict], original: list[dict]) -> None:
    out = ["# Older model run inventory", "",
           "Generated CPU-only record. Read [the narrative](OLDER_MODEL_HISTORY.md) for the reason for each revision.", "",
           "Old `training`/`evaluating` states can be stale. This inventory does not treat them as live processes.", "",
           "## Guarded run, get-up and diagnostic launches", ""]
    for row in records:
        path = Path(row["run"])
        state = row["state"]
        out.extend([f"### {path.name}", "", link(path, "Preserved folder"), "",
                    f"Saved status: `{state.get('status', 'not recorded')}`; stage: `{state.get('stage', 'not recorded')}`; "
                    f"recorded additional iterations: `{state.get('completed_additional_iterations', state.get('completed_iterations', state.get('total_training_iterations', 'not recorded')))}`."])
        if state.get("message"):
            out.append("Saved reason: " + flat(state["message"]))
        for k in ("last_accepted_checkpoint", "final_checkpoint", "checkpoint", "resume_checkpoint"):
            if state.get(k):
                out.append(f"- `{k}`: {link(state[k])}")
        gate = state.get("last_gate", {})
        if gate:
            out.append("- Latest saved gate fields: " + "; ".join(f"`{k}={gate[k]}`" for k in (
                "passed", "walking_pass", "stage_pass", "minimum_success", "minimum_height",
                "unsafe_fraction", "gate", "start_mode") if k in gate))
        for name in row["gates"]:
            out.append(f"- {link(path / name)}")
        if row["blocks"]:
            out.extend(["", "Saved blocks/probes (including failed or missing-result launches):", ""])
            for block in row["blocks"]:
                folder = Path(block["folder"])
                result = block["result"]
                ev = block["evaluation"]
                detail = "; ".join(f"{k}={result[k]}" for k in ("completed", "iterations", "seconds", "num_envs") if k in result)
                if ev:
                    detail += "; " + "; ".join(f"{k}={ev[k]}" for k in (
                        "strict_successes", "success_rate", "unsafe_terminated", "max_height_mean") if k in ev)
                out.append(f"- {link(folder)}: {detail or 'No parsed result/evaluation; see preserved files.'}")
        out.append("")
    out.extend(["## Original RSL-RL sessions", "",
                "These include configuration smokes, interrupted sessions and pre-axis-correction artifacts.",
                "A highest checkpoint number is a save index, not proof of qualification or unique trained updates.", "",
                "| Task/session | Root checkpoint files | Highest numbered root save |",
                "| --- | ---: | --- |"])
    for row in original:
        path = Path(row["run"])
        last = row["last_root_checkpoint"]
        out.append(f"| {link(path, path.parent.name + '/' + path.name)} | {row['root_checkpoint_count']} | "
                   f"{link(last) if last else 'None recorded'} |")
    (HISTORY / "OLDER_RUNS.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def write_evidence_index(diagnostics, checks, cad, reports) -> None:
    out = ["# Evidence and CAD archive index", "",
           "Generated from preserved local files. Full path/size/timestamp inventories are in RUN_INVENTORY.json.", "",
           "Reports can contain dated 'active' statements; ACTIVE_RUN.json and newer handoffs take precedence.",
           "CPU FK images, synthetic trajectories, actual Isaac screenshots and learned-policy evaluations have different scopes.", "",
           "## Training reports", ""]
    for record in reports:
        out.append("- " + link(record["path"]))
    for title, records in (("Diagnostics", diagnostics), ("Checks and controlled experiments", checks),
                           ("Physical CAD archives", cad)):
        out.extend(["", "## " + title, ""])
        for row in records:
            folder = Path(row["folder"])
            out.extend(["### " + folder.name, "", link(folder, "Evidence folder"), ""])
            files = row["files"]
            docs = [p for p in files if Path(p["path"]).suffix.lower() in (".md", ".html")]
            important = [p for p in files if Path(p["path"]).suffix.lower() == ".json"
                         and Path(p["path"]).parent == folder]
            for item in docs + important:
                out.append("- " + link(item["path"]))
            out.append(f"\nCatalogued evidence/authoring files: {len(files)}. Nested records and failed trials are retained in the machine inventory.\n")
    (HISTORY / "EVIDENCE_INDEX.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    HISTORY.mkdir(exist_ok=True)
    physical = [inspect_physical(p) for p in sorted((TRAINING / "runs").glob("locomotion_*"), key=run_number)
                if p.is_dir()]
    by_path = {row["run"].lower(): row for row in physical}
    for index, row in enumerate(physical):
        config = row["chain_config"]
        parent = next((config[k] for k in ("resume_run", "migrate_run", "reward_migrate_run") if config.get(k)), None)
        comparison = by_path.get(str(Path(parent)).replace("\\", "/").lower()) if parent else None
        basis = "declared parent" if comparison else "previous numbered run; comparison only, not proven actor lineage"
        comparison = comparison or (physical[index - 1] if index else None)
        if comparison:
            old = comparison["chain_config"].get("sources_sha256", {})
            new = config.get("sources_sha256", {})
            row["manifest_difference"] = {"compared_to": comparison["revision"], "basis": basis,
                "changed": sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
                "added": sorted(new.keys() - old.keys()), "removed": sorted(old.keys() - new.keys())}
    legacy = [inspect_legacy(p) for p in sorted((OLD / "logs").iterdir()) if p.is_dir() and
              (p.name.startswith(("guarded_", "getup_")) or (p / "state.json").exists())]
    original = [inspect_legacy(p) for p in sorted((OLD / "logs/rsl_rl").glob("*/*")) if p.is_dir()]
    diagnostics = evidence_tree(TRAINING / "diagnostics")
    checks = evidence_tree(TRAINING / "checks")
    cad = evidence_tree(CAD, ("cad_mesh_env", "__pycache__"))
    reports = [file_record(p) for p in sorted(TRAINING.glob("*.md"))]
    payload = {"schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Read-only retrospective inventory, not live qualification; no simulator/process/automation mutations",
               "active_pointer_at_inventory": read_json(TRAINING / "ACTIVE_RUN.json"),
               "physical_runs": physical, "legacy_runs": legacy,
               "original_rsl_sessions": original, "diagnostics": diagnostics, "checks": checks,
               "cad_archives": cad, "training_reports": reports, "read_warnings": WARNINGS}
    (HISTORY / "RUN_INVENTORY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_physical_details(physical)
    write_legacy_details(legacy, original)
    write_evidence_index(diagnostics, checks, cad, reports)
    print(json.dumps({"physical_runs": len(physical), "legacy_launches": len(legacy),
                      "original_rsl_sessions": len(original), "cad_archive_groups": len(cad),
                      "physical_operations": sum(len(p["operations"]) for p in physical),
                      "missing_manifest_artifacts": sum(p["actual_sha256"] is None for r in physical for p in r["frozen_source_verification"]),
                      "differing_manifest_hashes": sum(p["actual_sha256"] is not None and not p["matches"] for r in physical for p in r["frozen_source_verification"]),
                      "read_warnings": WARNINGS}, indent=2))


if __name__ == "__main__":
    main()
