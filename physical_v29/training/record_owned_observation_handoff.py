"""CPU-only provenance/status snapshot for the active physical-v29 chain.

Does not import Isaac, torch, or change any run, policy, gate or source snapshot.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import psutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics/walk_regression_20261003'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def processes():
    found = []
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'create_time', 'cmdline']):
        try:
            info = proc.info
            if (info['name'] or '').lower() not in ('python.exe', 'pythonw.exe'):
                continue
            args = info['cmdline'] or []
            # Match actual script arguments, not command text from a CPU audit.
            if any(Path(arg).name in ('worker_unitree.py', 'supervisor_unitree.py') for arg in args):
                found.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def owns_output(args, run):
    """Match the exact output path across Windows slash/case spellings."""
    try:
        output = Path(args[args.index('--output') + 1]).resolve()
        root = run.resolve()
        return output == root or root in output.parents
    except (ValueError, IndexError, OSError):
        return False


def main():
    active = read(HERE / 'ACTIVE_RUN.json')
    run = Path(active['run'])
    assert run.resolve().parent == (HERE/'runs').resolve(), 'Active run is outside the owned training directory'
    frozen = run / 'source/physical_v29'
    config = read(run / 'chain_config.json')
    integrity = {}
    for name, expected in config['sources_sha256'].items():
        if name in ('supervisor_unitree.py', 'vram_ramp.py'):
            path = run / name
        elif name.startswith('physical_assets/'):
            path = frozen / 'assets' / name.removeprefix('physical_assets/')
        elif name.startswith('physical_v29/'):
            path = frozen / name.removeprefix('physical_v29/')
        else:
            path = frozen / 'training' / name
        integrity[name] = hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert all(integrity.values()), integrity

    # Re-read at an operation boundary if the supervisor changed child workers.
    for _ in range(3):
        state = read(run / 'state.json')
        owned = processes()
        owned_here = [p for p in owned if owns_output(p['cmdline'] or [], run)]
        if not state.get('child_pid') or any(p['pid'] == state['child_pid'] for p in owned_here):
            break
        time.sleep(1)
    supervisor_live = any(p['pid'] == state['supervisor_pid'] for p in owned_here)
    if state['status'] in ('paused_for_performance_review', 'failed'):
        assert not supervisor_live and not owned_here, 'Non-running chain still owns a training process'
        assert not state.get('child_pid'), 'Non-running chain retains a child ownership claim'
        assert Path(state['checkpoint']).is_file(), 'Non-running checkpoint is missing'
    else:
        assert supervisor_live, 'Supervisor ownership unverified'
    workers = [p for p in owned if any(Path(a).name == 'worker_unitree.py' for a in (p['cmdline'] or []))]
    leaves = sorted({p['pid'] for p in workers} - {p['ppid'] for p in workers})
    assert len(leaves) <= 1, 'More than one actual Isaac/CUDA worker'
    assert not workers or all(owns_output(p['cmdline'] or [], run) for p in workers), 'Another run owns a worker'

    operations = []
    for folder in sorted((p for p in run.iterdir() if p.is_dir() and p.name.split('_', 1)[0].isdigit()),
                         key=lambda p: (int(p.name.split('_', 1)[0]), p.name)):
        item = dict(folder=str(folder))
        for name in ['result.json', 'warm_start.json', 'run_config.json']:
            if (folder / name).exists():
                item[name.removesuffix('.json')] = read(folder / name)
        if (folder / 'progress.jsonl').exists():
            rows = [json.loads(x) for x in (folder / 'progress.jsonl').read_text().splitlines()]
            if rows:
                item['updates'] = len(rows)
                item['latest_progress'] = rows[-1]
                item['ppo_summary'] = dict(
                    pre_update_kl_max=max(r['ppo']['pre_update_kl'] for r in rows),
                    pre_update_kl_reset_max=max((r['ppo']['pre_update_kl_reset'] or 0.) for r in rows),
                    first_post_update_kl=rows[0]['ppo']['post_update_kl'],
                    latest_post_update_kl=rows[-1]['ppo']['post_update_kl'],
                    minimum_vram_free_mb=min(r['gpu']['free_mb'] for r in rows),
                    minimum_host_available_mb=min(r['gpu']['ram_available_mb'] for r in rows),
                    raw_environment_steps_per_second=statistics.mean(
                        r['num_envs'] * 24 / (r['collect_seconds'] + r['learn_seconds']) for r in rows),
                )
        operations.append(item)

    current = Path(state['operation_path']) if state.get('operation_path') else None
    progress = read(current / 'progress.json') if current and (current / 'progress.json').exists() else None
    bench = read(run / 'benchmarks.json') if (run / 'benchmarks.json').exists() else []
    by_count = {b['environments']: b for b in bench}
    comparison = None
    if 6144 in by_count and 8192 in by_count:
        a, b = by_count[6144], by_count[8192]
        comparison = dict(raw_8192_over_6144=b['environment_steps_per_second'] / a['environment_steps_per_second'],
            effective_8192_over_6144=b['effective_environment_steps_per_second'] / a['effective_environment_steps_per_second'],
            scope='Bounded capacity tests, including startup/exit and any checkpoint recycling; not long-run convergence.')
    mem = psutil.virtual_memory()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader'], text=True, timeout=10).strip()
    seed_checkpoint = active.get('initial_full_checkpoint') or active.get('initial_actor_checkpoint')
    assert seed_checkpoint and Path(seed_checkpoint).is_file(), 'Missing initialization checkpoint provenance'
    reused = bool(state.get('migrated_from') and state.get('promotions'))
    value = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), recorded_unix=time.time(),
        active_run=active, state=state, state_age_seconds=time.time()-state['updated_unix'],
        processes=owned_here, actual_worker_pids=leaves, frozen_source_integrity=integrity,
        worker_progress=progress, gpu=gpu,
        system_memory=dict(available_mb=mem.available // 2**20, total_mb=mem.total // 2**20, used_percent=mem.percent),
        benchmarks=bench, throughput_comparison=comparison, operations=operations,
        seed_checkpoint_sha256=hashlib.sha256(Path(seed_checkpoint).read_bytes()).hexdigest(),
        qualification_reused=reused,
        qualification_reuse_scope='Matching-runtime previously qualified skills only; current and retention gates still mandatory' if reused else 'None',
        scope='Evidence snapshot; diagnostic and capacity updates do not count as qualified locomotion. Read matching full gates.')
    OUT.mkdir(exist_ok=True)
    save(OUT / 'live_handoff.json', value)
    print(json.dumps(dict(recorded_utc=value['recorded_utc'], status=state['status'], stage=state['stage'],
        operation_path=state.get('operation_path'), environments=state.get('environment_count'),
        next_training_environment_count=state.get('next_training_environment_count'),
        full_training_iterations=state['total_training_iterations'], actual_worker_pids=leaves,
        gpu=gpu, ram_available_mb=value['system_memory']['available_mb'],
        frozen_files_verified=len(integrity), benchmarks=bench, throughput_comparison=comparison,
        current_iteration=progress.get('iteration') if progress else None,
        current_ppo=progress.get('ppo') if progress else None,
        promotions=state['promotions']), indent=2))


if __name__ == '__main__':
    main()
