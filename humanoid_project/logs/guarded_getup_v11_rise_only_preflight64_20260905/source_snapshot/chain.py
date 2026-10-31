"""Rise-only v11 discovery supervisor for 5-foot humanoid recovery.

This permissive experiment asks only whether PPO can raise the pelvis from one
supine reset family and then four lying reset families. It discovers upward
motion; it does not certify a stable, natural, or hardware-safe get-up.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
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
parser.add_argument('--preflight_only', action='store_true')
args = parser.parse_args()

OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'state.json').exists():
    raise RuntimeError('Choose a new run directory; completed or interrupted runs are immutable')


PLAN = [
    # First reproduce upward escape from one supine family, then expose all
    # four lying families. These stages judge only attained pelvis height.
    dict(name='supine_rise', start_mode='lying', groups=1,
         stage='discovery', budget=800, eval_seconds=12.,
         minimum_height=.75, perturb=False, optional=False),
    dict(name='four_pose_rise', start_mode='lying', groups=4,
         stage='discovery', budget=1200, eval_seconds=12.,
         minimum_height=.75, perturb=False, optional=False),
]
BASE_BUDGET = sum(item['budget'] for item in PLAN)
HARD_CEILING = 2000
BLOCK = 200

state = dict(status='initializing', supervisor_pid=os.getpid(), child_pid=0,
             requested_num_envs=args.num_envs, num_envs=args.num_envs,
             reserve_mb=args.reserve_mb, plan=PLAN, completed_iterations=0,
             base_budget=BASE_BUDGET, total_budget=HARD_CEILING,
             history=[], latest_checkpoint=None, accepted_checkpoint=None,
             message='')


def save(status, **updates):
    state.update(status=status, updated=datetime.now().isoformat(), **updates)
    temporary = OUT / 'state.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(OUT / 'state.json')
    print(f'[{state["updated"]}] {status}: {state.get("message", "")}', flush=True)


def gpu_sample():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('nvidia-smi telemetry unavailable: ' + result.stderr.strip())
    values = [part.strip() for part in result.stdout.splitlines()[0].split(',')]
    used, total = int(values[0]), int(values[1])
    return dict(used_mb=used, total_mb=total, free_mb=total-used,
                utilization=int(values[2]), power_w=values[3])


class MemoryPressure(RuntimeError):
    pass


def terminate_owned(process):
    if process.poll() is None:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, timeout=30)
        process.wait(timeout=20)


def run_worker(label, mode, checkpoint=None, *, count=64, iterations=0,
               groups=1, group=0, seed=47, perturb=False,
               stage='discovery', start_mode='lying', eval_seconds=20.,
               learning_rate=3e-4):
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-u', str(HERE / 'worker.py'),
               '--mode', mode, '--stage', stage, '--output', str(folder),
               '--num_envs', str(count), '--iterations', str(iterations),
               '--groups', str(groups), '--group', str(group), '--seed', str(seed),
               '--start_mode', start_mode, '--eval_seconds', str(eval_seconds),
               '--learning_rate', str(learning_rate), '--headless']
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    if perturb:
        command += ['--perturb']
    (folder / 'command.json').write_text(json.dumps(command, indent=2))

    peak_used = 0
    low_free_samples = 0
    result_seen_at = None
    with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        save('training' if mode == 'train' else 'evaluating', child_pid=process.pid,
             active_block=label, active_output=str(folder),
             block_iterations=iterations if mode == 'train' else 0,
             message=f'{label}; PID {process.pid}')
        try:
            while process.poll() is None:
                try:
                    sample = gpu_sample()
                    peak_used = max(peak_used, sample['used_mb'])
                    with (OUT / 'gpu.csv').open('a', newline='') as stream:
                        csv.writer(stream).writerow(
                            [datetime.now().isoformat(), label, process.pid, *sample.values()])
                    low_free_samples = low_free_samples + 1 if sample['free_mb'] < args.reserve_mb else 0
                    if low_free_samples >= 3:
                        raise MemoryPressure(
                            f'GPU free memory stayed below {args.reserve_mb} MiB for three samples')
                except MemoryPressure:
                    raise
                except (OSError, ValueError, RuntimeError) as telemetry_error:
                    save(state['status'], telemetry_warning=str(telemetry_error),
                         message=f'{label}; training continues while GPU telemetry retries')
                if (folder / 'result.json').exists():
                    if result_seen_at is None:
                        result_seen_at = time.monotonic()
                    elif time.monotonic() - result_seen_at > 60:
                        terminate_owned(process)
                        break
                time.sleep(5)
        except BaseException:
            terminate_owned(process)
            raise

    result_file = folder / 'result.json'
    if not result_file.exists():
        stdout_tail = (folder / 'stdout.log').read_text(errors='replace')[-5000:]
        stderr_tail = (folder / 'stderr.log').read_text(errors='replace')[-5000:]
        detail = stdout_tail + '\n' + stderr_tail
        lowered = detail.lower()
        if any(token in lowered for token in
               ('out of memory', 'cuda error: memory allocation', 'bad allocation')):
            raise MemoryPressure(detail[-2500:])
        raise RuntimeError(f'{label} failed (exit {process.returncode}): {detail[-3500:]}')
    result = json.loads(result_file.read_text())
    if not result.get('completed'):
        raise RuntimeError(f'{label} produced an incomplete result')
    if mode == 'train' and not Path(result['checkpoint']).is_file():
        raise RuntimeError(f'{label} did not produce its promised checkpoint')
    result['peak_sampled_gpu_mb'] = peak_used
    state['history'].append(dict(block=label, mode=mode, result=result))
    save('between_blocks', child_pid=0, message=f'{label} finished')
    return result


def fallback_count(current):
    candidates = [n for n in (8192, 6144, 4096) if n < current]
    return candidates[0] if candidates else None


def train_block(label, checkpoint, spec, iterations, learning_rate):
    retry = 0
    while True:
        count = state['num_envs']
        suffix = f'_{count}' + (f'_retry{retry}' if retry else '')
        try:
            return run_worker(label + suffix, 'train', checkpoint, count=count,
                              iterations=iterations, groups=spec['groups'],
                              stage=spec['stage'], start_mode=spec['start_mode'],
                              learning_rate=learning_rate)
        except MemoryPressure as error:
            lower = fallback_count(count)
            if lower is None:
                raise
            save('memory_fallback', num_envs=lower,
                 message=f'{error}; retrying the same checkpoint at {lower} environments')
            retry = 0
        except RuntimeError as error:
            if retry >= 1:
                raise
            retry += 1
            save('worker_retry', message=f'{label} failed once; clean retry from the prior checkpoint: {error}')


evaluation_sequence = 0


def evaluate_checkpoint(tag, checkpoint, spec, allow_perturb=True):
    global evaluation_sequence
    evaluation_sequence += 1
    prefix = f'e{evaluation_sequence:03d}_{tag}'
    nominal = []
    for group in range(spec['groups']):
        result = run_worker(f'{prefix}_g{group}_nominal', 'eval', checkpoint,
                            count=64, groups=spec['groups'], group=group,
                            seed=1049 + group, stage=spec['stage'],
                            start_mode=spec['start_mode'],
                            eval_seconds=spec['eval_seconds'])
        nominal.append(result['report'])

    # Deliberately do not use stable-standing, contact, penetration, drift, or
    # posture gates for discovery. The only promotion signal is whether every
    # evaluated pose family raises its pelvis to the requested height.
    nominal_required = 0.
    nominal_ok = all(report['max_height_mean'] >= spec['minimum_height']
                     for report in nominal)
    perturbed = []
    if nominal_ok and spec.get('perturb', False) and allow_perturb:
        for group in range(spec['groups']):
            result = run_worker(f'{prefix}_g{group}_perturbed', 'eval', checkpoint,
                                count=64, groups=spec['groups'], group=group,
                                seed=2089 + group, perturb=True,
                                stage=spec['stage'], start_mode=spec['start_mode'],
                                eval_seconds=25.)
            perturbed.append(result['report'])
    perturb_required = 0.
    perturb_ok = (not spec.get('perturb', False) or not allow_perturb
                  or (len(perturbed) == spec['groups']
                      and all(report['success_rate'] >= perturb_required
                              and report['peak_penetration_m'] <= .015
                              for report in perturbed)))
    reports = nominal + perturbed
    unsafe_fraction = max((report['unsafe_terminated'] / report['trials'] for report in reports),
                          default=1.)
    minimum_success = min((report['success_rate'] for report in reports), default=0.)
    minimum_height = min((report['max_height_mean'] for report in nominal), default=0.)
    gate = dict(passed=nominal_ok and perturb_ok, checkpoint=str(checkpoint),
                start_mode=spec['start_mode'], groups=spec['groups'],
                discovery_only=True, minimum_height_required=spec['minimum_height'],
                nominal_required=nominal_required,
                perturbed_required=perturb_required if spec.get('perturb', False) else None,
                minimum_success=minimum_success, minimum_height=minimum_height,
                unsafe_fraction=unsafe_fraction, reports=reports)
    (OUT / f'{prefix}_gate.json').write_text(json.dumps(gate, indent=2))
    save('gate_result', last_gate=gate,
         message=(f'{spec["name"]}: diagnostic stable success {minimum_success:.1%}, '
                  f'mean-height floor {minimum_height:.3f} m; '
                  f'passed={gate["passed"]}'))
    return gate


def gate_score(gate):
    return min(gate['minimum_height'] / gate['minimum_height_required'], 1.)


def run_unit_contracts():
    transcript = []
    for name in ('test_reward_core.py', 'test_evaluation.py', 'test_revision.py'):
        result = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE,
                                capture_output=True, text=True, timeout=120)
        transcript.append(f'===== {name} =====\n{result.stdout}\n{result.stderr}')
        if result.returncode:
            (OUT / 'contract_tests.log').write_text('\n'.join(transcript))
            raise RuntimeError(f'{name} failed; no GPU training was started')
    (OUT / 'contract_tests.log').write_text('\n'.join(transcript))


def snapshot_inputs():
    snapshot = OUT / 'source_snapshot'
    snapshot.mkdir()
    inputs = []
    for source in HERE.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
        inputs.append(source)
    inputs += [HERE / 'assets/recovery.usda', HERE / 'assets/pose_bank.pt',
               ROOT / 'assets/urdf/humanoid_5ft_isaac.urdf']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (OUT / 'input_hashes.json').write_text(json.dumps(hashes, indent=2))


def preflight():
    save('preflight', message='Running code contracts and full-size rise-only allocation')
    run_unit_contracts()
    anchor = run_worker('p00_zero_controller_anchor', 'train', count=64,
                        iterations=0, groups=1, stage='discovery', start_mode='lying')
    checkpoint = anchor['checkpoint']
    while True:
        count = state['num_envs']
        try:
            benchmark = run_worker(f'p02_benchmark_{count}', 'train', checkpoint,
                                   count=count, iterations=4, groups=1,
                                   stage='discovery', start_mode='crouch')
            break
        except MemoryPressure as error:
            lower = fallback_count(count)
            if lower is None:
                raise
            save('memory_fallback', num_envs=lower,
                 message=f'Preflight memory fallback: {error}')
    record = dict(anchor_checkpoint=checkpoint, selected_num_envs=state['num_envs'],
                  benchmark_seconds=benchmark['seconds'],
                  benchmark_peak_gpu_mb=benchmark['peak_sampled_gpu_mb'],
                  discovery_only=True)
    (OUT / 'preflight.json').write_text(json.dumps(record, indent=2))
    save('preflight_complete', latest_checkpoint=checkpoint,
         accepted_checkpoint=checkpoint,
         message=(f'Checkpoint reload passed; {state["num_envs"]} environments allocated; '
                  f'peak sampled GPU memory {benchmark["peak_sampled_gpu_mb"]} MiB'))
    return checkpoint


def main():
    if not (HERE / 'assets/pose_audit.json').is_file():
        raise RuntimeError('The independently completed pose audit is missing')
    snapshot_inputs()
    with (OUT / 'gpu.csv').open('w', newline='') as stream:
        csv.writer(stream).writerow(
            ['timestamp', 'block', 'pid', 'used_mb', 'total_mb', 'free_mb',
             'utilization', 'power_w'])
    checkpoint = preflight()
    if args.preflight_only:
        return

    accepted = checkpoint
    for stage_index, spec in enumerate(PLAN):
        state.update(stage=spec['name'], stage_index=stage_index,
                     stage_budget=spec['budget'], stage_completed=0)
        save('stage_start', latest_checkpoint=checkpoint,
             accepted_checkpoint=accepted,
             message=(f'{spec["name"]}: endpoint {spec["start_mode"]}, '
                      f'{spec["budget"]} planned rise-only iterations'))

        gate = evaluate_checkpoint(f'{stage_index:02d}_{spec["name"]}_baseline',
                                   checkpoint, spec)
        if gate['passed']:
            accepted = checkpoint
            save('stage_skipped', accepted_checkpoint=accepted,
                 message=f'{spec["name"]} already passes independently; no training spent')
            continue

        continuation = checkpoint
        best_checkpoint = checkpoint
        best_score = gate_score(gate)
        used = 0
        regression_count = 0
        learning_rate = 1e-4 if spec['stage'] == 'discovery' else 5e-5
        stage_limit = spec['budget']
        passed = False
        while used < stage_limit and state['completed_iterations'] < HARD_CEILING:
            iterations = min(BLOCK, stage_limit-used,
                             HARD_CEILING-state['completed_iterations'])
            if iterations <= 0:
                break
            label = f'{stage_index:02d}_{spec["name"]}_{used+iterations:04d}'
            trained = train_block(label, continuation, spec, iterations, learning_rate)
            candidate = trained['checkpoint']
            used += iterations
            state['completed_iterations'] += iterations
            state['stage_completed'] = used
            state['latest_checkpoint'] = candidate
            gate = evaluate_checkpoint(label, candidate, spec)
            score = gate_score(gate)
            if score > best_score + .01:
                best_score, best_checkpoint = score, candidate
                regression_count = 0
            elif score < best_score - .25:
                regression_count += 1
            else:
                regression_count = 0

            if gate['passed']:
                checkpoint = candidate
                accepted = candidate
                state['accepted_checkpoint'] = accepted
                passed = True
                save('stage_passed', message=(f'{spec["name"]} passed after {used} iterations; '
                                              'advancing rise discovery'))
                break

            if regression_count >= 2:
                continuation = best_checkpoint
                learning_rate = max(learning_rate * .5, 5e-5)
                regression_count = 0
                save('rollback', continuation_checkpoint=continuation,
                     message=(f'{spec["name"]} regressed twice; rolling back to the measured '
                              f'best candidate and reducing LR to {learning_rate:g}'))
            else:
                continuation = candidate
            checkpoint = continuation

        if passed:
            continue
        if spec.get('optional', False):
            checkpoint = accepted
            save('optional_refine_rejected', latest_checkpoint=checkpoint,
                 accepted_checkpoint=accepted,
                 message='Optional refinement rejected; retaining the last height-passing policy')
            continue
        save('incomplete_stage', child_pid=0, latest_checkpoint=best_checkpoint,
             accepted_checkpoint=accepted,
             message=(f'{spec["name"]} did not reach the height target within its budget. '
                      'The best checkpoint is preserved; harder stages were not falsely started.'))
        return

    save('completed', child_pid=0, final_checkpoint=checkpoint,
         latest_checkpoint=checkpoint, accepted_checkpoint=accepted,
         message='All rise-only height stages passed; this is not a stable-standing certification')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save('failed', child_pid=0, message=str(error))
        raise
