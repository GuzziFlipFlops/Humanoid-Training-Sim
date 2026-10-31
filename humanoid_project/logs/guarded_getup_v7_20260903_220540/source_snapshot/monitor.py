"""Read-only recovery monitor; never changes/restarts training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, default=None)
parser.add_argument('--once', action='store_true')
args = parser.parse_args()


def duration(seconds):
    if seconds is None: return 'pending measurement'
    seconds = max(0, round(seconds))
    return f'{seconds//3600}h {(seconds%3600)//60}m {seconds%60}s'


def show():
    runs = sorted(ROOT.glob('logs/guarded_getup_v7_*/state.json'), key=lambda p: p.stat().st_mtime)
    file = args.run / 'state.json' if args.run else runs[-1] if runs else None
    if file is None:
        print('No recovery chain state yet; setup/preflight may still be in progress.', flush=True)
        return False
    state = json.loads(file.read_text())
    active = state.get('active_output')
    iteration, target, sec = None, None, None
    if active and (Path(active)/'stdout.log').exists():
        with (Path(active)/'stdout.log').open('rb') as stream:
            stream.seek(0, 2); length = stream.tell(); stream.seek(max(0, length-200000))
            log = stream.read().decode(errors='replace')
        found = re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', log)
        if found: iteration, target = map(int, found[-1]); iteration += 1
        timings = re.findall(r'collection:\s*([\d.]+)s,\s*learning:\s*([\d.]+)s', log)
        if timings:
            recent = [float(a)+float(b) for a, b in timings[-10:]]
            sec = sum(recent)/len(recent)
        else:
            recent = re.findall(r'Iteration time:\s*([\d.]+)s', log)
            if recent: sec = sum(map(float, recent[-10:]))/len(recent[-10:])
    print('\n' + datetime.now().strftime('%H:%M:%S') + f" | {state['status']} | {state.get('stage','preflight')}")
    active_count = state['num_envs']
    if active and (Path(active)/'command.json').exists():
        command = json.loads((Path(active)/'command.json').read_text())
        active_count = command[command.index('--num_envs')+1]
    print(f"Active environments: {active_count} | Selected training count: {state['num_envs']} | GPU reserve: {state['reserve_mb']} MiB")
    print(state.get('active_block','setup'))
    if state['status'] == 'training' and iteration is not None:
        print(f'Current block: {iteration}/{target} iterations | ' +
              (f'{1/sec:.3f} iter/s ({sec:.2f} s/iter)' if sec else 'timing not available'))
        print('ETA to next evaluation: ' + duration((target-iteration)*sec if sec else None))
        stage_done = state.get('stage_completed', 0) + iteration
        remaining = max(0, state.get('stage_budget', target)-stage_done)
        print(f"Stage: {stage_done}/{state.get('stage_budget',target)} maximum iterations | training-only ETA: " + duration(remaining*sec if sec else None))
    elif state['status'] == 'evaluating':
        print('Evaluating a saved policy; PPO iteration count is not advancing.')
    else:
        print('Main completed iterations:', state['completed_iterations'], '/', state['total_budget'], 'maximum')
    gate = state.get('last_gate')
    if gate:
        if 'minimum_success' in gate:
            print(f"Last independent worst-pose success: {gate['minimum_success']:.1%} | gate passed: {gate['passed']}")
        else:
            report = gate.get('report', {})
            print(f"Last assisted diagnostic height: {report.get('max_height_mean', 0.):.3f} m | "
                  f"promotion eligible: {gate.get('promotion_eligible', False)} | gate passed: {gate['passed']}")
    else:
        print('Independent get-up success: not measured yet (training reward is not success).')
    if state.get('message'): print(state['message'])
    print('Policy:', state.get('latest_checkpoint') or 'no completed policy yet', flush=True)
    return state['status'] in ('completed', 'failed', 'preflight_complete') or state['status'].startswith('stopped_')


while True:
    try: stopped = show()
    except (OSError, ValueError) as e:
        print('Waiting for readable state:', e, flush=True); stopped = False
    if args.once or stopped: break
    time.sleep(10)
