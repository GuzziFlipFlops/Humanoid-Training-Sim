"""Start one source-frozen repaired chain after the previous one is drained."""
import argparse,json,subprocess,time
from pathlib import Path
import psutil
from status_io import atomic_json

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--prior_run',required=True);p.add_argument('--name',required=True)
p.add_argument('--resource_only',action='store_true',help='Continue the latest same-reward walking actor/critic/Adam')
p.add_argument('--reward_migration',action='store_true',help='Keep latest learned actor under repaired rewards and independently requalify prior skills')
p.add_argument('--gate_repair',action='store_true',help='Explicit audited v2-to-v3 landing measurement repair; all prior gates re-run')
args=p.parse_args()
assert not (args.resource_only and args.reward_migration)
assert not args.gate_repair or args.reward_migration
prior=Path(args.prior_run).resolve();assert prior.is_relative_to(here/'runs')
old=json.loads((prior/'state.json').read_text())
assert old['status']=='superseded_for_diagnostic_repair' and old['child_pid'] is None
for proc in psutil.process_iter(['name','cmdline']):
    command=' '.join(proc.info['cmdline'] or []).replace('\\','/')
    assert not ('python' in (proc.info['name'] or '').lower() and
        any(k in command for k in ('training/worker.py','training/supervisor.py'))),command
assert '/' not in args.name and '\\' not in args.name
run=here/'runs'/args.name;run.mkdir(exist_ok=True);assert not (run/'state.json').exists()
initial=here/'checks/reward_repair_20261001/stand_normalization_repaired.pt'
if args.resource_only:
    from supervisor import verify_resource_migration
    verify_resource_migration(prior,here.parent)
    initial=Path(old['checkpoint']);assert initial.exists()
if args.reward_migration:
    from supervisor import verify_reward_migration
    verify_reward_migration(prior,here.parent,args.gate_repair)
    initial=Path(old['checkpoint']);assert initial.exists()
command=[str(here.parents[1]/'env_isaaclab/Scripts/python.exe'),'-u','-X','utf8',str(here/'supervisor.py'),
    '--output',str(run),'--reserve_mb','300','--max_hours','18','--iterations_per_block','64',
    '--validated_checks',str(here/'checks/reward_repair_20261001/reused_checks'),
    '--initial_checkpoint',str(initial),'--train_envs','8192']
if args.resource_only:command+=['--migrate_run',str(prior)]
if args.reward_migration:command+=['--reward_migrate_run',str(prior)]
if args.gate_repair:command+=['--gate_repair']
with (run/'supervisor_stdout.log').open('w',encoding='utf-8') as out,(run/'supervisor_stderr.log').open('w',encoding='utf-8') as err:
    proc=subprocess.Popen(command,stdout=out,stderr=err,cwd=str(here.parents[1]),creationflags=subprocess.CREATE_NO_WINDOW)
atomic_json(run/'launch.json',dict(command=command,pid=proc.pid,time=time.time(),prior_run=str(prior),
    source_checkpoint=str(initial),target_environments=8192,reason='Gait integrity plus training-only stopped-command regularization to verified standing prior; keep learned actor, reset values only for different rewards, requalify prior skills' if args.reward_migration else 'Host memory maintenance; preserve learned actor/critic/Adam and adaptive learning rate' if args.resource_only else 'Event fall penalty, balance preservation, bounded reference and independent running hop verification',
    diagnosis=str(here/'checks/landing_audit_20261002_1245/diagnosis.json')))
atomic_json(here/'ACTIVE_RUN.json',dict(run=str(run),supervisor_pid=proc.pid,started_unix=time.time(),prior_run=str(prior),
    note='Gait integrity v6 and training-only standing prior regularization; v3 independent gates requalify all skills; target 8192, reserve 300 MiB, ramp 180s' if args.reward_migration else 'Balance-event v4, physical landing verifier, RAM guard and checkpoint-preserving recycling; target 8192, reserve 300 MiB, ramp 180s'))
print(json.dumps(dict(run=str(run),launcher_pid=proc.pid)))
