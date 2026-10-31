"""Bounded, logged get-up discovery -> pose diversity -> refinement supervisor."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=8192)
parser.add_argument('--reserve_mb', type=int, default=512)
parser.add_argument('--resume_checkpoint', type=Path, required=True,
    help='Earlier rising v2 policy; actor mean/normalizer migrate, critic/optimizer/noise reset')
parser.add_argument('--preflight_from', type=Path, default=None)
parser.add_argument('--preflight_only', action='store_true')
parser.add_argument('--smoke_from', type=Path, default=None,
    help='Reuse a manually inspected completed 64-env/64-iteration smoke; reload/scaling are still required')
parser.add_argument('--hold_from', type=Path, default=None,
    help='Reuse a manually inspected hold-controller calibration for the unchanged asset/criteria')
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'state.json').exists(): raise RuntimeError('Choose a new run directory; do not overwrite an existing state')
PLAN = [('supine_discovery', 1, 600), ('four_pose_discovery', 4, 800), ('refine', 4, 400)]
state = dict(status='initializing', supervisor_pid=os.getpid(), child_pid=0, num_envs=args.num_envs,
             reserve_mb=args.reserve_mb, plan=PLAN, completed_iterations=0, total_budget=1800,
             inherited_iterations=400, inherited_checkpoint=str(args.resume_checkpoint.resolve()),
             history=[], best_checkpoint=None, latest_checkpoint=None)


def save(status, **kw):
    state.update(status=status, updated=datetime.now().isoformat(), **kw)
    tmp = OUT / 'state.tmp'
    tmp.write_text(json.dumps(state, indent=2)); tmp.replace(OUT / 'state.json')
    print(f'[{state["updated"]}] {status}: {kw.get("message", "")}', flush=True)


def gpu():
    p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if p.returncode: raise RuntimeError('GPU telemetry unavailable')
    v = [x.strip() for x in p.stdout.splitlines()[0].split(',')]
    return dict(used_mb=int(v[0]), total_mb=int(v[1]), free_mb=int(v[1])-int(v[0]), utilization=int(v[2]), power_w=v[3])


class MemoryPressure(RuntimeError): pass


def terminate_owned(p):
    if p.poll() is None:
        subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True, timeout=30)
        p.wait(timeout=20)


def worker(label, mode, checkpoint=None, count=64, iterations=64, groups=1, group=0, seed=47, perturb=False, stage='discovery'):
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-u', str(HERE/'worker.py'), '--mode', mode, '--stage', stage,
               '--output', str(folder), '--num_envs', str(count), '--iterations', str(iterations),
               '--groups', str(groups), '--group', str(group), '--seed', str(seed), '--headless']
    if checkpoint: command += ['--checkpoint', str(checkpoint)]
    if perturb: command += ['--perturb']
    (folder/'command.json').write_text(json.dumps(command, indent=2))
    peak, consecutive, completed_at = 0, 0, None
    with (folder/'stdout.log').open('w') as stdout, (folder/'stderr.log').open('w') as stderr:
        p = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        save('training' if mode == 'train' else 'evaluating', child_pid=p.pid, active_block=label,
             active_output=str(folder), block_iterations=iterations if mode == 'train' else 0,
             message=f'{label}, PID {p.pid}')
        try:
            while p.poll() is None:
                sample = gpu()
                peak = max(peak, sample['used_mb'])
                with (OUT/'gpu.csv').open('a', newline='') as stream:
                    csv.writer(stream).writerow([datetime.now().isoformat(), label, p.pid, *sample.values()])
                consecutive = consecutive + 1 if sample['free_mb'] < args.reserve_mb else 0
                if consecutive >= 3: raise MemoryPressure('Free GPU memory stayed below reserve for three samples')
                if (folder/'result.json').exists():
                    if completed_at is None: completed_at = time.monotonic()
                    elif time.monotonic()-completed_at > 60:
                        terminate_owned(p)
                        break
                time.sleep(5)
        except BaseException:
            terminate_owned(p)
            raise
    result_file = folder/'result.json'
    if not result_file.exists():
        error = (folder/'stderr.log').read_text(errors='replace')
        if any(word in error.lower() for word in ('out of memory', 'cuda error: memory allocation', 'bad allocation')):
            raise MemoryPressure(error[-1500:])
        raise RuntimeError(f'{label} failed (exit {p.returncode}): {error[-2000:]}')
    result = json.loads(result_file.read_text())
    if not result.get('completed'): raise RuntimeError('Missing completed result')
    if mode == 'train' and not Path(result['checkpoint']).is_file(): raise RuntimeError('Missing completed checkpoint')
    result['peak_sampled_gpu_mb'] = peak
    state['history'].append(dict(block=label, mode=mode, result=result))
    save('between_blocks', child_pid=0, message=label + ' finished')
    return result


def evaluations(label, checkpoint, groups, stage):
    rows = []
    for group in range(groups):
        for seed, perturb in ((47, False), (83, True)):
            result = worker(f'{label}_g{group}_s{seed}', 'eval', checkpoint, count=128,
                            groups=groups, group=group, seed=seed, perturb=perturb, stage=stage)
            rows.append(result['report'])
    passing = all(r['success_rate'] >= (.85 if r['perturb'] else .90)
                  and r['peak_penetration_m'] <= .015 and r['unsafe_terminated'] == 0 for r in rows)
    gate = dict(passed=passing, minimum_success=min(r['success_rate'] for r in rows), reports=rows,
                checkpoint=str(checkpoint), per_group_required=True)
    (OUT/(label+'_gate.json')).write_text(json.dumps(gate, indent=2))
    return gate


def train(label, checkpoint, groups, stage):
    while True:
        count = state['num_envs']
        try:
            return worker(f'{label}_{count}', 'train', checkpoint, count=count, iterations=200, groups=groups, stage=stage)
        except MemoryPressure as exc:
            lower = next((n for n in (8192, 6144, 4096) if n < count), None)
            if lower is None: raise
            save('memory_fallback', num_envs=lower, message=f'{exc}; retry this block at {lower}')


def main():
    if not (HERE/'assets/pose_audit.json').exists(): raise RuntimeError('No completed pose audit')
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for source in HERE.glob('*.py'): shutil.copy2(source, snapshot/source.name)
    files = [ROOT/'humanoid_cfg.py', ROOT/'source/humanoid_tasks/curriculum/env_cfg.py',
             ROOT/'assets/usd/humanoid_5ft.usd',
             ROOT/'logs/guarded_run_v4_20260903_0305/06_speed_4_0600_8192/candidate.pt',
             HERE/'assets/recovery.usda', HERE/'assets/pose_bank.pt', args.resume_checkpoint.resolve()]
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (OUT/'input_hashes.json').write_text(json.dumps(original, indent=2))
    with (OUT/'gpu.csv').open('w', newline='') as stream:
        csv.writer(stream).writerow(['timestamp', 'block', 'pid', 'used_mb', 'total_mb', 'free_mb', 'utilization', 'power_w'])
    save('preflight', message='Small train, saved-policy reload, memory/throughput checks')
    if args.preflight_from:
        pre = json.loads((args.preflight_from/'preflight.json').read_text())
        if pre['input_hashes'] != original: raise RuntimeError('Preflight inputs differ')
        if pre.get('source_hashes') != code_hashes: raise RuntimeError('Preflight source changed; rerun checks')
        checkpoint = pre['checkpoint']
        baseline = pre['baseline']
        state['num_envs'] = pre['num_envs']
    else:
        if args.smoke_from:
            smoke = json.loads((args.smoke_from/'result.json').read_text())
            if not (smoke.get('completed') and smoke['num_envs'] == 64 and smoke['iterations'] >= 64
                    and smoke['groups'] == 4 and smoke['action_scale'] == .7):
                raise RuntimeError('Supplied smoke does not match the required recovery smoke')
            save('preflight', reused_smoke=str(args.smoke_from), message='Reusing inspected small smoke; still testing reload and scaling')
        else:
            smoke = worker('smoke64', 'train', args.resume_checkpoint, count=64, iterations=64, groups=4)
        checkpoint = smoke['checkpoint']
        if args.hold_from:
            hold = json.loads((args.hold_from/'result.json').read_text())
            if not (hold.get('completed') and hold['report']['group'] == 4 and hold['report']['trials'] >= 16):
                raise RuntimeError('Invalid supplied standing-hold calibration')
            save('preflight', reused_hold=str(args.hold_from), message='Reusing inspected stationary feasibility test')
        else:
            hold = worker('standing_hold_calibration', 'hold', count=16, groups=4, group=4)
        if hold['report']['strict_successes'] == 0:
            raise RuntimeError('No audited standing pose satisfied the final hold; inspect telemetry before training')
        reload = worker('smoke_reload', 'eval', checkpoint, count=16, groups=4, group=0, perturb=True)
        # There is no accepted recovery predecessor to compare against. Reuse
        # the reload probe as an initial signal only, NOT as a promotion gate.
        baseline = dict(minimum_success=reload['report']['success_rate'], reports=[reload['report']])
        checked = []
        for count in (4096, 6144, 8192):
            if count > args.num_envs: continue
            try:
                r = worker('benchmark_' + str(count), 'train', checkpoint, count=count, iterations=4, groups=4)
                checked.append((count, r['seconds']/4., r['peak_sampled_gpu_mb']))
            except MemoryPressure:
                break
        if not checked: raise RuntimeError('No full-size count passed preflight')
        state['num_envs'] = checked[-1][0]
        state['benchmarks'] = checked
    (OUT/'preflight.json').write_text(json.dumps(dict(input_hashes=original, checkpoint=checkpoint,
        source_hashes=code_hashes, baseline=baseline,
        num_envs=state['num_envs'], benchmarks=state.get('benchmarks', [])), indent=2))
    if args.preflight_only:
        save('preflight_complete', child_pid=0, message='Smoke, checkpoint reload and scaling checks passed')
        return
    # Smoke/benchmark updates are discarded: main work branches from exactly
    # the preserved 200-iteration actor, not the 64-env preflight candidate.
    checkpoint = str(args.resume_checkpoint.resolve())
    state['latest_checkpoint'] = checkpoint
    retained_groups = set()
    for stage_index, (name, groups, budget) in enumerate(PLAN):
        stage = 'refine' if name == 'refine' else 'discovery'
        save('stage_start', stage=name, stage_index=stage_index, stage_completed=0, stage_budget=budget,
             message=f'{name}; cap {budget} iterations, measured gates every 200')
        best_score = baseline['minimum_success'] if stage_index == 0 else -1.
        best_height = max(r['max_height_mean'] for r in baseline['reports']) if stage_index == 0 else 0.
        no_progress = 0
        for completed in range(0, budget, 200):
            label = f'{stage_index:02d}_{name}_{completed+200:04d}'
            result = train(label, checkpoint, groups, stage)
            candidate = result['checkpoint']
            state['completed_iterations'] += 200
            state['latest_checkpoint'] = candidate
            state['stage_completed'] = completed + 200
            gate = evaluations(label, candidate, groups, stage)
            retention_ok = all(r['success_rate'] >= (.80 if r['perturb'] else .85)
                               for r in gate['reports'] if r['group'] in retained_groups)
            # After adding poses, include supine again: no promotion at its expense.
            score = gate['minimum_success']
            height = max(r['max_height_mean'] for r in gate['reports'])
            improved = score > best_score + .005 or (best_score == 0 and height > best_height + .02)
            no_progress = 0 if improved else no_progress + 1
            if retention_ok and score >= best_score - .02:
                checkpoint = candidate  # discovery continuation, NOT an approved/passing policy
            best_score, best_height = max(best_score, score), max(best_height, height)
            save('gate_result', last_gate=gate, latest_checkpoint=candidate, continuation_checkpoint=checkpoint,
                 message=f'{name}: worst-case success {score:.1%}; passed={gate["passed"]}')
            if any(r['unsafe_terminated'] / r['trials'] >= .90 for r in gate['reports']):
                save('stopped_unsafe', child_pid=0,
                     message='At least 90% unsafe trials: height progress cannot justify further blind training')
                return
            if gate['passed']:
                state['best_checkpoint'] = candidate
                retained_groups.update(range(groups))
                checkpoint = candidate
                break
            if no_progress >= 3:
                save('stopped_no_progress', child_pid=0, message='Three evaluated blocks without true success/height progress; preserve candidate for diagnosis')
                return
        else:
            save('stopped_gate', child_pid=0, message=f'{name} budget reached without required per-pose success; not promoted')
            return
    save('completed', final_checkpoint=checkpoint, child_pid=0, message='All recovery/hold gates passed')


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failed', child_pid=0, message=str(error))
        raise
