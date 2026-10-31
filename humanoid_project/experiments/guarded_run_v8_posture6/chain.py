"""Local-only guarded fine tuning. Never overwrites or promotes archived policies.

Owns its child processes, GPU log, tested checkpoints and rejection records.
Failure to meet a gate stops advancement; a time budget never overrides a gate.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from gates import judge, proven_velocity_limited
from resume import accepted_source

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WALK=ROOT/'logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt'
RUN=ROOT/'logs/rsl_rl/humanoid5ft_natural_run_from_working_walk/2026-09-01_19-54-34/model_3800.pt'
STAGES=[('speed',4.5,400),('speed',5.0,500),('speed',5.5,600),('speed',6.0,800)]
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--reserve_mb',type=int,default=512)
parser.add_argument('--num_envs',type=int,default=8192)
parser.add_argument('--preflight_only',action='store_true')
parser.add_argument('--validated_preflight_from',type=str,default=None)
parser.add_argument('--resume_from',required=True,help='Evidence-preserving guarded-v7 state.json with isolated posture failure')
args=parser.parse_args()
OUT=Path(args.output).resolve()
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'state.json').exists(): raise RuntimeError('Choose a new output directory; archived states are not overwritten')
STATE={'supervisor_pid':os.getpid(),'status':'initializing','child_pid':0,'num_envs':args.num_envs,
       'reserve_mb':args.reserve_mb,'plan':STAGES,'max_additional_iterations':sum(s[2] for s in STAGES),
       'approved_walker':str(WALK),'original_runner':str(RUN),'history':[],
       'velocity_boost':1.0,'boost_classes':[]}


def save(status,**kwargs):
    STATE.update(status=status,timestamp=datetime.now().isoformat(),**kwargs)
    path=OUT/'state.tmp'
    path.write_text(json.dumps(STATE,indent=2),encoding='utf-8'); path.replace(OUT/'state.json')
    print(f'[{datetime.now().isoformat(timespec="seconds")}] {status}: {kwargs.get("message", "")}',flush=True)


def gpu():
    proc=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if proc.returncode: raise RuntimeError('GPU telemetry unavailable')
    a=[s.strip() for s in proc.stdout.splitlines()[0].split(',')]
    return dict(used_mb=int(a[0]),total_mb=int(a[1]),free_mb=int(a[1])-int(a[0]),
                utilization=int(a[2]),power_w=a[3],temperature_c=a[4])


class MemoryPressure(RuntimeError): pass


def stop_owned_worker(proc):
    # Windows venv python.exe can be a redirector with the actual simulator as
    # a child. Killing only the redirector leaks the GPU process into the retry.
    if proc.poll() is None:
        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,timeout=30)
        proc.wait(timeout=20)


def worker(label,mode,checkpoint,scale,stage='index',ceiling=1.8,count=64,iterations=3,seed=47,
           perturb=False,resume_optimizer=False,learning_rate=5e-5,
           velocity_boost=1.0,boost_classes=()):
    folder=OUT/label; folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u',str(HERE/'worker.py'),'--mode',mode,'--stage',stage,'--ceiling',str(ceiling),
             '--scale',str(scale),'--checkpoint',str(checkpoint),'--reference',str(WALK),
             '--output',str(folder),'--num_envs',str(count),'--iterations',str(iterations),
             '--seed',str(seed),'--learning_rate',str(learning_rate),
             '--velocity_boost',str(velocity_boost),'--boost_classes',','.join(boost_classes),'--headless']
    if perturb: command.append('--perturb')
    if resume_optimizer: command.append('--resume_optimizer')
    (folder/'command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
    started=time.monotonic(); low=0; marker_at=None; max_used=0
    with (folder/'stdout.log').open('w',encoding='utf-8') as stdout, (folder/'stderr.log').open('w',encoding='utf-8') as stderr:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        save('training' if mode=='train' else 'evaluating',child_pid=proc.pid,active_block=label,
             stdout=str(folder/'stdout.log'),stderr=str(folder/'stderr.log'),message=label,
             block_iteration=0,block_iterations=iterations,child_elapsed_s=0.)
        try:
            while proc.poll() is None:
                time.sleep(10)
                if proc.poll() is not None: break
                sample=gpu(); max_used=max(max_used,sample['used_mb'])
                with (OUT/'gpu.csv').open('a',newline='',encoding='utf-8') as f:
                    csv.writer(f).writerow([datetime.now().isoformat(),label,proc.pid,*sample.values()])
                STATE['gpu']=sample; STATE['child_elapsed_s']=round(time.monotonic()-started,1)
                if mode=='train':
                    with (folder/'stdout.log').open('rb') as progress:
                        progress.seek(max(0,(folder/'stdout.log').stat().st_size-24000))
                        matches=re.findall(r'Learning iteration\s+(\d+)/(\d+)',progress.read().decode('utf-8',errors='replace'))
                    if matches:
                        STATE['block_iteration']=int(matches[-1][0])+1
                        STATE['block_iterations']=int(matches[-1][1])
                        STATE['stage_iterations_display']=STATE.get('stage_completed_before_block',0)+STATE['block_iteration']
                # Write state without producing 10-second chatty console logs.
                temp=OUT/'state.tmp'; temp.write_text(json.dumps(STATE,indent=2),encoding='utf-8'); temp.replace(OUT/'state.json')
                low=low+1 if sample['free_mb']<args.reserve_mb else 0
                if low>=3: raise MemoryPressure('Three samples below GPU reserve')
                if (folder/'result.json').exists():
                    marker_at=marker_at or time.monotonic()
                    if time.monotonic()-marker_at>60:
                        # Only our worker, after a durable completed result: simulator teardown hung.
                        stop_owned_worker(proc); break
                last=max((folder/'stdout.log').stat().st_mtime,(folder/'stderr.log').stat().st_mtime)
                if time.time()-last>900: raise RuntimeError('No worker output for 15 minutes')
        except BaseException:
            stop_owned_worker(proc)
            raise
    result=folder/'result.json'
    if not result.exists():
        errors=(folder/'stderr.log').read_text(encoding='utf-8',errors='replace')
        if any(text in errors.lower() for text in ('out of memory','cuda error: memory allocation','bad allocation')):
            raise MemoryPressure(errors[-1500:])
        raise RuntimeError(f'{label} failed (exit {proc.returncode}); inspect {folder}/stderr.log: {errors[-1500:]}')
    value=json.loads(result.read_text(encoding='utf-8'))
    if not value.get('completed'): raise RuntimeError('Incomplete worker result')
    value['peak_sampled_gpu_mb']=max_used
    STATE['history'].append(dict(block=label,mode=mode,result=value))
    save('between_blocks',child_pid=0,message=label+' completed')
    return value


def evaluations(label,checkpoint,scale,stage,ceiling,velocity_boost=1.0,boost_classes=()):
    reports=[]
    for seed,perturb in ((47,False),(83,True)):
        value=worker(f'{label}_eval_{seed}','eval',checkpoint,scale,stage,ceiling,seed=seed,perturb=perturb,
                     velocity_boost=velocity_boost,boost_classes=boost_classes)
        reports.append(json.loads(Path(value['evaluation']).read_text()))
    return reports


def train_block(label,checkpoint,scale,stage,ceiling,iterations,resume_optimizer=False,learning_rate=5e-5,
                velocity_boost=1.0,boost_classes=()):
    while True:
        count=STATE['num_envs']
        try:
            return worker(f'{label}_{count}','train',checkpoint,scale,stage,ceiling,count,iterations,
                          resume_optimizer=resume_optimizer,learning_rate=learning_rate,
                          velocity_boost=velocity_boost,boost_classes=boost_classes)
        except MemoryPressure as error:
            lower=next((n for n in (8192,6144,4096) if n<count),None)
            if lower is None: raise
            save('memory_fallback',num_envs=lower,message=f'{error}; retry only this block from its input checkpoint at {lower}')


def main():
    with (OUT/'gpu.csv').open('w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(['timestamp','block','pid','used_mb','total_mb','free_mb','utilization','power_w','temperature_c'])
    snapshots=OUT/'source_snapshot'; snapshots.mkdir()
    for p in HERE.glob('*.py'): shutil.copy2(p,snapshots/p.name)
    original=[WALK,RUN,ROOT/'humanoid_cfg.py',ROOT/'source/humanoid_tasks/curriculum/env_cfg.py',
              ROOT/'source/humanoid_tasks/curriculum/skill_mdp.py',ROOT/'assets/usd/humanoid_5ft.usd']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    (OUT/'original_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    STATE['original_hashes']=hashes
    checkpoint,scale,source_gate=accepted_source(args.resume_from,hashes)
    seed_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (OUT/'input_checkpoint.json').write_text(json.dumps({
        'checkpoint':str(checkpoint),'sha256':seed_digest,'source_state':args.resume_from,
        'policy_note':'Immutable v6 4 m/s checkpoint that passed both full gates; rejected 4.5 m/s candidates are not resumed.'},indent=2))
    STATE['resume_gate']=source_gate
    STATE['resume_source']=args.resume_from
    STATE['resume_checkpoint']=str(checkpoint)
    save('preflight',message='Small posture-reward smoke, checkpoint reload and 8192-environment check')
    smoke=worker('smoke64','train',checkpoint,scale,'speed',4.5,64,16)
    worker('smoke_checkpoint_reload','eval',smoke['checkpoint'],scale,'speed',4.5)
    benchmarks=[]
    counts=tuple(dict.fromkeys(n for n in (args.num_envs,6144,4096) if n<=args.num_envs))
    for count in counts:
        if count>args.num_envs: continue
        try:
            b=worker(f'benchmark_{count}','train',checkpoint,scale,'speed',4.5,count,4)
            b['environment_steps_per_second']=count*24*4/b['seconds']; benchmarks.append(b)
            break
        except MemoryPressure as error:
            STATE['history'].append(dict(block=f'benchmark_{count}',memory_failure=str(error)))
            continue
    if not benchmarks: raise RuntimeError('No environment count passed preflight')
    STATE['num_envs']=benchmarks[-1]['num_envs']
    STATE['benchmarks']=benchmarks
    if args.preflight_only:
        save('preflight_complete',message='No main training requested by this invocation'); return
    # The evaluator now records per-joint saturation and a stricter upright
    # cone, so baseline results are measured fresh rather than copied from v4.
    seed_reports=evaluations('walking_seed',WALK,.35,'index',1.0)
    if not judge(seed_reports,'index')['walking_pass']:
        raise RuntimeError('Approved walker fails its matched self-check; investigate evaluator before training')
    resume_reports=evaluations('resume_check',checkpoint,scale,'speed',4.5)
    if not judge(resume_reports,'index')['walking_pass']:
        raise RuntimeError('Resume candidate failed fresh walking retention tests')
    save('seed_selected',seed_checkpoint=str(checkpoint),action_scale=scale,
         message='Warm start chosen by measured walking retention, not training return')
    STATE['last_accepted_checkpoint']=str(checkpoint)
    cached=(str(checkpoint),4.5,1.0,tuple(),resume_reports,'resume_check')
    total=0
    for stage_idx,(stage,ceiling,budget) in enumerate(STAGES):
        STATE.update(stage=stage,stage_index=stage_idx,speed_ceiling=ceiling,stage_budget=budget)
        accepted=Path(checkpoint); stage_passed=False; lr=2.5e-5; same_stage_optimizer=False
        STATE['stage_completed_before_block']=0
        block_size=100 if stage in ('targets','torso','traction') else 200
        STATE['block_size']=block_size
        velocity_boost=STATE['velocity_boost']
        boost_classes=tuple(STATE['boost_classes'])
        # A fresh measured pass is sufficient; training is a means, not a quota.
        prelabel=f'{stage_idx:02d}_{stage}_{ceiling:g}_0000'
        cache_key=(str(accepted),ceiling,velocity_boost,boost_classes)
        if cached[:4]==cache_key:
            reports=cached[4]; evaluated_as=cached[5]
        else:
            reports=evaluations(prelabel,accepted,scale,stage,ceiling,velocity_boost,boost_classes)
            evaluated_as=prelabel
            cached=cache_key+(reports,prelabel)
        precheck=judge(reports,stage,seed_reports)
        precheck.update(checkpoint=str(accepted),stage=stage,ceiling=ceiling,
                        evaluated_as=evaluated_as,pretraining=True)
        (OUT/(prelabel+'_gate.json')).write_text(json.dumps(precheck,indent=2))
        STATE['last_gate']=precheck
        if precheck['walking_pass'] and precheck['stage_pass']:
            save('stage_passed',message=f'{stage} at {ceiling:g} m/s already passes; no extra training required')
            continue
        for offset in range(0,budget,block_size):
            STATE['stage_completed_before_block']=offset
            STATE['block_iteration']=0
            iterations=min(block_size,budget-offset)
            label=f'{stage_idx:02d}_{stage}_{ceiling:g}_{offset+iterations:04d}'
            block=train_block(label,accepted,scale,stage,ceiling,iterations,same_stage_optimizer,lr,
                              velocity_boost,boost_classes)
            total+=iterations; STATE['completed_additional_iterations']=total
            candidate=Path(block['checkpoint'])
            reports=evaluations(label,candidate,scale,stage,ceiling,velocity_boost,boost_classes)
            result=judge(reports,stage,seed_reports)
            result.update(checkpoint=str(candidate),input_checkpoint=str(accepted),stage=stage,ceiling=ceiling,evaluated_as=label)
            (OUT/(label+'_gate.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
            STATE['last_gate']=result
            if not result['walking_pass']:
                # Never continue from a walking-regressed update; retry the accepted state more gently.
                lr=max(2.5e-5,lr*.5); same_stage_optimizer=False
                save('candidate_rejected',message=f'{label}: walking regression; rollback to previous accepted checkpoint')
                continue
            accepted=candidate; same_stage_optimizer=True
            cached=(str(accepted),ceiling,velocity_boost,boost_classes,reports,label)
            STATE['last_accepted_checkpoint']=str(accepted)
            if result['stage_pass']:
                stage_passed=True
                save('stage_passed',message=f'{stage} at {ceiling:g} m/s passed both test batches')
                break
            save('stage_not_yet_passed',message=f'{label}: walking retained, stage quality not yet sufficient')
        # A one-time velocity increase is permitted only when both stable test
        # batches prove a speed deficit plus sustained saturation.  Falling or
        # merely approaching a limit is not sufficient evidence.
        eligible=proven_velocity_limited(reports) if not stage_passed else []
        if not stage_passed and velocity_boost == 1.0 and eligible:
            velocity_boost=1.15; boost_classes=tuple(eligible)
            STATE.update(velocity_boost=velocity_boost,boost_classes=list(boost_classes),
                         velocity_boost_evidence={
                             'stage':stage,'ceiling':ceiling,'classes':list(boost_classes),
                             'source_evaluations':[r['seed'] for r in reports],
                             'criterion':'>10% at >=95% limit in both stable, speed-deficient batches'})
            bonus_budget=400
            STATE['stage_budget']=budget+bonus_budget
            same_stage_optimizer=False
            save('velocity_limit_proven',message=(
                f'{ceiling:g} m/s is measurably velocity-limited in {", ".join(boost_classes)}; '
                'applying the single allowed 15% isolated-candidate boost'))
            for bonus_offset in range(0,bonus_budget,200):
                STATE['stage_completed_before_block']=budget+bonus_offset
                iterations=min(200,bonus_budget-bonus_offset)
                label=f'{stage_idx:02d}_{stage}_{ceiling:g}_v115_{bonus_offset+iterations:04d}'
                block=train_block(label,accepted,scale,stage,ceiling,iterations,same_stage_optimizer,lr,
                                  velocity_boost,boost_classes)
                total+=iterations; STATE['completed_additional_iterations']=total
                candidate=Path(block['checkpoint'])
                reports=evaluations(label,candidate,scale,stage,ceiling,velocity_boost,boost_classes)
                result=judge(reports,stage,seed_reports)
                result.update(checkpoint=str(candidate),input_checkpoint=str(accepted),stage=stage,
                              ceiling=ceiling,evaluated_as=label,velocity_boost=velocity_boost,
                              boost_classes=list(boost_classes))
                (OUT/(label+'_gate.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
                STATE['last_gate']=result
                if not result['walking_pass']:
                    lr=max(2.5e-5,lr*.5); same_stage_optimizer=False
                    save('candidate_rejected',message=f'{label}: walking regression; rollback')
                    continue
                accepted=candidate; same_stage_optimizer=True
                cached=(str(accepted),ceiling,velocity_boost,boost_classes,reports,label)
                STATE['last_accepted_checkpoint']=str(accepted)
                if result['stage_pass']:
                    stage_passed=True
                    save('stage_passed',message=f'{stage} at {ceiling:g} m/s passed after measured 15% boost')
                    break
        if not stage_passed:
            save('stopped_gate',message=f'{stage} did not pass within {budget} iterations. Speed NOT advanced.',
                 last_accepted_checkpoint=str(accepted))
            return
        checkpoint=accepted
    save('completed',message='All planned measured gates passed; final checkpoint still needs human visual review',
         final_checkpoint=str(checkpoint))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        save('failed',message=str(error),child_pid=0)
        raise
    finally:
        if 'original_hashes' in STATE:
            check={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in STATE['original_hashes'].items()}
            (OUT/'originals_preserved.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
