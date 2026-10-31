"""CPU-only snapshot and exact-checkpoint gate inventory for this dated check."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import psutil

OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record():
    active = read(HERE / 'ACTIVE_RUN.json')
    run = Path(active['run'])
    state = read(run / 'state.json')
    config = read(run / 'chain_config.json')
    hashes = {}
    for name, expected in config['sources_sha256'].items():
        frozen = run / 'source/physical_v29'
        if name in ('supervisor_unitree.py', 'vram_ramp.py'):
            path = run / name
        elif name.startswith('physical_assets/'):
            path = frozen / 'assets' / name.split('/', 1)[1]
        elif name.startswith('physical_v29/'):
            path = frozen / name.split('/', 1)[1]
        else:
            path = frozen / 'training' / name
        hashes[name] = dict(expected=expected, actual=digest(path))
        assert hashes[name]['actual'] == expected, name

    owned = []
    for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'create_time']):
        try:
            info = proc.info
            args = info['cmdline'] or []
            if (info['name'] or '').lower() == 'python.exe' and any(
                    Path(a).name in ('worker_unitree.py', 'supervisor_unitree.py') for a in args):
                output_arg = next((args[i+1] for i,a in enumerate(args[:-1]) if a=='--output'), None)
                assert output_arg is not None and (Path(output_arg).resolve()==run.resolve() or run.resolve() in Path(output_arg).resolve().parents), 'Another run owns a simulator process'
                owned.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert any(p['pid'] == state['supervisor_pid'] for p in owned)
    workers = [p for p in owned if any(Path(a).name == 'worker_unitree.py' for a in p['cmdline'])]
    actual_workers = sorted({p['pid'] for p in workers} - {p['ppid'] for p in workers})
    assert len(actual_workers) <= 1

    gates = []
    for file in sorted(run.glob('*_gate_*/result.json')):
        result = read(file)
        if not result.get('completed'):
            continue
        cfg = read(file.parent / 'run_config.json')
        checkpoint = Path(cfg['arguments']['checkpoint'])
        gates.append(dict(folder=str(file.parent), checkpoint=str(checkpoint),
                          checkpoint_sha256=digest(checkpoint),
                          run_config=cfg, result=result))
    qualifications = []
    for promotion in state['promotions']:
        if Path(promotion['checkpoint']).parent.parent != run:
            qualifications.append(dict(promotion=promotion, scope='Previously qualified exact checkpoint; preserved historical evidence'))
            continue
        matches = [g for g in gates if g['checkpoint'] == promotion['checkpoint']
                   and abs(g['run_config']['arguments']['speed'] - promotion['speed']) < 1e-6
                   and g['result']['passed']]
        nominal = next(g for g in matches if not g['result']['perturb'])
        pushed = next(g for g in matches if g['result']['perturb'])
        assert nominal['checkpoint_sha256'] == pushed['checkpoint_sha256']
        assert all(c['statistics']['trials'] >= 32 for g in (nominal, pushed) for c in g['result']['cases'])
        qualifications.append(dict(promotion=promotion, checkpoint_sha256=nominal['checkpoint_sha256'],
                                   nominal=nominal['folder'], pushed=pushed['folder'],
                                   nominal_cases=nominal['result']['cases'], pushed_cases=pushed['result']['cases'],
                                   scope='Matching full nominal/pushed/stopped/prior-command gates, >=32 robots per command'))

    blocks = []
    for attempt in state['attempts']:
        folder = Path(attempt['folder'])
        if attempt['operation'] != 'train' or '_train_' not in folder.name:
            continue
        result_path = folder / 'result.json'
        if not result_path.exists():
            continue
        result = read(result_path)
        rows = [json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
        n = attempt['environments']
        samples = n * 24 * result['iterations']
        active_time = sum(r['collect_seconds'] + r['learn_seconds'] for r in rows)
        blocks.append(dict(folder=str(folder), speed=attempt['speed'], updates=result['iterations'],
                           environments=n, samples=samples, wall_seconds=attempt['seconds'],
                           active_seconds=active_time, active_PPO_samples_s=samples/active_time,
                           worker_effective_samples_s=samples/attempt['seconds'],
                           minimum_free_vram_mb=min(r['gpu']['free_mb'] for r in rows),
                           minimum_host_available_mb=min(r['gpu']['ram_available_mb'] for r in rows),
                           first_private_mb=rows[0]['gpu']['process_private_mb'], last_private_mb=rows[-1]['gpu']['process_private_mb'],
                           first_torch_reserved_mb=rows[0]['gpu']['torch_reserved_mb'], last_torch_reserved_mb=rows[-1]['gpu']['torch_reserved_mb'],
                           max_pre_update_KL=max(r['ppo']['pre_update_kl'] for r in rows),
                           max_action_command_difference=max(r['behavior']['action_command_difference'] for r in rows),
                           recycled=result.get('recycled', False)))
    operation = Path(state['operation_path'])
    progress = read(operation/'progress.json') if (operation/'progress.json').exists() else None
    now = time.time()
    completed_samples = sum(b['samples'] for b in blocks)
    partial_samples = 0
    if state.get('operation') == 'train' and (operation/'progress.jsonl').exists() and not (operation/'result.json').exists():
        partial_samples = state['environment_count'] * 24 * len((operation/'progress.jsonl').read_text().splitlines())
    baseline_path = OUT/'preceding_check.json'
    if not baseline_path.exists():
        if (OUT/'evidence.json').exists():
            previous = read(OUT/'evidence.json')['preceding_check']
        else:
            previous = read(HERE/'automation/NEXT_CHECK.json')
        baseline_path.write_text(json.dumps(previous,indent=2)+'\n',encoding='utf-8')
    previous = read(baseline_path)
    baseline_time = datetime.fromisoformat(previous['recorded_utc']).timestamp()
    newer_samples = sum(b['samples'] for b in blocks if (Path(b['folder'])/'result.json').stat().st_mtime > baseline_time)
    # A check may occur mid-block. Interval numerator labels this boundary explicitly.
    evidence = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), active_pointer=active,
                    state=state, state_age_seconds=now-state['updated_unix'], frozen_hashes=hashes,
                    processes=owned, actual_worker_pids=actual_workers, gates=gates, qualifications=qualifications,
                    completed_main_blocks=blocks, current_worker_progress=progress,
                    current_checkpoint_sha256=digest(Path(state['checkpoint'])),
                    completed_main_samples=completed_samples, partial_main_samples=partial_samples,
                    full_chain_completed_main_samples_s=completed_samples/(now-state['started_unix']),
                    completed_main_active_PPO_samples_s=completed_samples/sum(b['active_seconds'] for b in blocks),
                    completed_main_worker_effective_samples_s=completed_samples/sum(b['wall_seconds'] for b in blocks),
                    preceding_check=previous,
                    interval_completed_block_samples=newer_samples,
                    interval_seconds=now-baseline_time,
                    interval_boundary_scope='Whole blocks completed since preceding check; excludes any unfinished block and can include its pre-check prefix. Use full-chain ratio for consistent denominator.',
                    gpu_smi=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip(),
                    host_available_mb=psutil.virtual_memory().available//2**20,
                    diagnostic_scope='CPU evidence only. No simulator, no source/policy/gate mutation. A prior qualified actor is distinct from the currently unqualified candidate.')
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(recorded_utc=evidence['recorded_utc'],stage=state['stage'],main_updates=state['total_training_iterations'],
                         operation=operation.name,qualified_speeds=[p['speed'] for p in state['promotions']],
                         completed_samples=completed_samples,active_PPO_samples_s=evidence['completed_main_active_PPO_samples_s'],
                         worker_effective_samples_s=evidence['completed_main_worker_effective_samples_s'],
                         full_chain_samples_s=evidence['full_chain_completed_main_samples_s'],
                         last_main_block=blocks[-1],actual_worker_pids=actual_workers),indent=2))


if __name__ == '__main__':
    record()
