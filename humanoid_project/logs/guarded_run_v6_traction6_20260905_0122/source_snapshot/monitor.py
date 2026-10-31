"""Read-only terminal monitor; no Isaac imports or training mutations."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
import time

LOG_ROOT=Path(r'E:\Aditya\robot_sim\humanoid_project\logs')
_RUNS=sorted(LOG_ROOT.glob('guarded_run_v6_traction6_*'),key=lambda p:p.stat().st_mtime)
DEFAULT=(_RUNS[-1]/'state.json') if _RUNS else (LOG_ROOT/'guarded_run_v6_traction6_PENDING'/'state.json')
TERMINAL={'completed','stopped_gate','failed','preflight_complete'}


def read_json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError): return None


def tail(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0,f.seek(0,2)-160000))
            return f.read().decode('utf-8',errors='replace')
    except OSError: return ''


def duration(seconds):
    if seconds is None or not math.isfinite(seconds): return 'estimating...'
    hours,seconds=divmod(max(0,round(seconds)),3600)
    minutes,seconds=divmod(seconds,60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else f'{minutes}m {seconds:02d}s'


def log_progress(text):
    iterations=re.findall(r'Learning iteration\s+(\d+)/(\d+)',text)
    times=[float(x) for x in re.findall(r'Iteration time:\s+([\d.]+)s',text)]
    return ((int(iterations[-1][0])+1,int(iterations[-1][1])) if iterations else (0,0),
            statistics.mean(times[-20:]) if times else None)


def eta_to(target,progress,block_size,rate,eval_pair,startup,pending_eval=0.):
    """To training iteration target; excludes that target's final evaluation."""
    if target<=progress: return 0.
    if rate is None: return None
    intermediate=sum(progress<b<target for b in range(block_size,target,block_size))
    return (target-progress)*rate+intermediate*(eval_pair+startup)+pending_eval


def timings(state,run):
    evaluation,startups,rates=[],[],[]
    for item in state.get('history',[]):
        r=item.get('result',{}); folder=run/item['block']
        try: wall=(folder/'result.json').stat().st_mtime-(folder/'command.json').stat().st_mtime+10.
        except OSError: wall=r.get('seconds',0.)+30.
        if item.get('mode')=='eval' and r.get('completed'): evaluation.append(wall)
        if item.get('mode')=='train' and r.get('completed'):
            startups.append(max(0.,wall-r['seconds']))
            if r.get('iterations',0)>=50: rates.append(r['seconds']/r['iterations'])
    return (statistics.median(evaluation[-6:]) if evaluation else 120.,
            statistics.median(startups[-4:]) if startups else 30.,rates[-1] if rates else None)


def success_lines(state,run):
    gate=state.get('last_gate')
    if gate:
        block=gate.get('evaluated_as',Path(gate['checkpoint']).parent.name.rsplit('_',1)[0])
        checks=[v for batch in gate['checks'] for group in ('walking','stage') for v in batch[group].values()]
        lines=[f'LATEST EVALUATED CHECKPOINT: {block}',
               f'Quality checks: {sum(checks)}/{len(checks)} passed (not a rollout success percentage)',
               f'Walking preserved: {"PASS" if gate["walking_pass"] else "FAIL"} | '
               f'Stage quality: {"PASS" if gate["stage_pass"] else "FAIL"}']
        failed=sorted({k for b in gate['checks'] for g in ('walking','stage') for k,v in b[g].items() if not v})
        if failed: lines.append('Failed checks: '+', '.join(failed))
    else:
        block='resume_check'
        lines=['LATEST EVALUATION: resumed checkpoint BEFORE new training',
               'New candidate quality: awaiting first completed checkpoint evaluation.']
    for seed,name in ((47,'Nominal'),(83,'Push test')):
        report=read_json(run/f'{block}_eval_{seed}'/'evaluation.json')
        if not report:
            lines.append(f'  {name}: no completed result yet'); continue
        for key,title in (('candidate_1.0','1 m/s'),('candidate_target','stage target')):
            if key=='candidate_target' and report['rows'][key]['target']==1.: continue
            r=report['rows'][key]; speed=r.get('mean_speed'); asym=r.get('stance_asymmetry')
            speed_text=f'{speed:.3f}' if speed is not None else 'n/a'
            asym_text=f'{asym:.3f}' if asym is not None else 'n/a'
            lines.append(f'  {name}, {title}: no-fall rate {100*(1-r["fall_fraction"]):.1f}% of {report["replicas"]} rollouts; '
                         f'speed {speed_text} m/s; stance imbalance {asym_text}')
            if key=='candidate_target' and r.get('torso_tilt_deg') is not None:
                lines.append(f'    Torso tilt {r["torso_tilt_deg"]:.1f} deg; '
                             f'lean-cone occupancy {100*r.get("torso_excess_fraction",0.):.1f}%')
        baseline=read_json(run/f'walking_seed_eval_{seed}'/'evaluation.json')
        if baseline and (state.get('stage')=='balance' or gate and gate.get('stage')=='balance'):
            before=baseline['rows']['candidate_1.0'].get('stance_asymmetry')
            after=report['rows']['candidate_1.0'].get('stance_asymmetry')
            if before and after is not None:
                lines.append(f'    Balance improvement: {100*(before-after)/before:+.1f}% '
                             f'(requires >=5%; imbalance <= {max(.10,.95*before):.3f})')
    lines.append('No-fall rate is NOT full task success. Results update only after evaluations.')
    return lines


def render(state,run):
    status=state['status']; stage=state.get('stage','preflight'); budget=state.get('stage_budget',0)
    block=state.get('active_block',''); block_size=state.get('block_size',100 if stage=='balance' else 200)
    eval_one,startup,prior_rate=timings(state,run)
    source=Path(state.get('stdout',run/'missing'))
    if status!='training' and stage!='preflight':
        prefix=re.sub(r'_eval_\d+$','',block)
        candidates=sorted(run.glob(prefix+'_[0-9]*/stdout.log'),key=lambda p:p.stat().st_mtime)
        if candidates: source=candidates[-1]
    (local,local_budget),rate=log_progress(tail(source)); rate=rate or prior_rate
    progress=min(budget,state.get('stage_completed_before_block',0)+local) if budget else local
    if status!='training':
        match=re.search(r'_(\d{4})_eval_\d+$',block)
        if match: progress=int(match[1])
    lines=[f'HUMANOID TRAINING | {datetime.now():%H:%M:%S}',
           f'Status: {status} | Stage: {stage} | Environments: {state["num_envs"]:,}',f'Active block: {block}']
    if budget: lines.append(f'Stage iterations attempted: {progress:,} / {budget:,} ({100*progress/budget:.1f}%)')
    if status=='training':
        lines.append(f'Current block: {local}/{local_budget}' if local_budget else 'Simulator starting; no iteration reported yet.')
    lines.append(f'Training speed: {1/rate:.3f} iterations/sec | {rate:.2f} sec/iteration (recent mean)' if rate
                 else 'Training speed: estimating after first iterations...')
    pending=0.
    if status=='training' and local_budget and local>=local_budget:
        # Last iteration logged, supervisor has not switched phase yet.
        pending=2*eval_one+startup
    if status=='evaluating':
        try: elapsed=time.time()-(run/block/'command.json').stat().st_mtime
        except OSError: elapsed=0.
        pending=max(0.,eval_one-elapsed)+(eval_one if block.endswith('_47') else 0.)+startup
        lines.append('Live training rate: 0 (evaluating); training-speed estimate above is retained.')
        lines.append(f'ETA until this evaluation pair/startup finishes: ~{duration(pending)} (can overrun)')
    if budget and status not in TERMINAL:
        next_check=min(budget,(progress//block_size+1)*block_size)
        milestone=min(budget,(progress//200+1)*200)
        def eta(target): return duration(eta_to(target,progress,block_size,rate,2*eval_one,startup,pending))
        if progress<budget:
            lines.extend([f'ETA to next checkpoint at iteration {next_check}: ~{eta(next_check)}',
                          f'ETA to next 200-iteration milestone ({milestone}): ~{eta(milestone)}',
                          f'ETA to all {budget} stage iterations: ~{eta(budget)}'])
            total=eta_to(budget,progress,block_size,rate,2*eval_one,startup,pending)
            lines.append(f'ETA to final stage evaluation: ~{duration(None if total is None else total+2*eval_one)}')
        else: lines.append('Stage iteration budget complete; awaiting quality decision.')
        plan=state.get('plan',[]); next_index=state.get('stage_index',-1)+1
        if 0<=next_index<len(plan):
            name,speed,iterations=plan[next_index]
            lines.append(f'Next stage: {name} up to {speed:g} m/s, max {iterations} iterations; starts ONLY after PASS.')
        lines.append(f'ETAs include intervening evaluations (~{duration(2*eval_one)}/pair) and startups.')
        lines.append('Assumes continuing through the budget; passing can shorten it, rejection can stop it.')
    elif status in TERMINAL: lines.append('No active-training ETA: the chain has stopped or finished.')
    lines.append(''); lines.extend(success_lines(state,run))
    sample=state.get('gpu')
    if sample:
        lines.extend(['',f'Last sampled GPU: {sample["utilization"]}% | VRAM {sample["used_mb"]:,}/{sample["total_mb"]:,} MiB '
                      f'| free {sample["free_mb"]:,} MiB | reserve {state["reserve_mb"]} MiB'])
    lines.extend([state.get('message',''),'Refresh every 5 sec. Ctrl+C stops ONLY this monitor.'])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--state',type=Path,default=DEFAULT)
    parser.add_argument('--once',action='store_true'); args=parser.parse_args()
    while True:
        state=read_json(args.state)
        if not args.once: print('\033[2J\033[H',end='')
        print(render(state,args.state.parent) if state else 'Waiting for readable state...',flush=True)
        if args.once or state and state['status'] in TERMINAL: return
        time.sleep(5)


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('\nMonitor closed. Training was not changed.')
