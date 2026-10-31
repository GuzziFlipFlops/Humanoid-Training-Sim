"""Dated report, local journal/history and next-check handoff from saved evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
e=json.loads((OUT/'evidence.json').read_text())
a=json.loads((OUT/'running_audit.json').read_text())
test=json.loads((OUT/'running_counterfactuals.json').read_text())
state=e['state']
run=Path(e['active_pointer']['run'])
gates=e['gates']
nom=next(g for g in reversed(gates) if not g['result']['perturb'] and any(
    p['result']['perturb'] and p['checkpoint_sha256']==g['checkpoint_sha256']
    and p['run_config']['arguments']['speed']==g['run_config']['arguments']['speed'] for p in gates))
push=next(g for g in reversed(gates) if g['result']['perturb']
          and g['checkpoint_sha256']==nom['checkpoint_sha256']
          and g['run_config']['arguments']['speed']==nom['run_config']['arguments']['speed'])
assert nom['checkpoint_sha256']==push['checkpoint_sha256'], 'Only matching completed gate pair may be summarized'
last=e['completed_main_blocks'][-1]
running_updates=sum(b['updates'] for b in e['completed_main_blocks'] if b['speed']==1.5)
qualified=[q for q in e['qualifications'] if q['promotion']['speed']==1.][0]
walk=next(c['statistics'] for c in qualified['nominal_cases'] if c['statistics']['command_m_s']==1.)
current={c['statistics']['command_m_s']:c for c in nom['result']['cases']}
motor=a['motor_audit']['commands'][-1]['motors']
ankles=[m for m in motor if 'ankle/' in m['name']]
max_ankle=max(m['maximum_cap_fraction'] for m in ankles)
min_vram=min(b['minimum_free_vram_mb'] for b in e['completed_main_blocks'])
min_host=min(b['minimum_host_available_mb'] for b in e['completed_main_blocks'])

table='| Command | Nominal signed speed | Pushed signed speed | Survival nominal/pushed | Failures |\n| --- | --- | --- | --- | --- |\n'
for n,p in zip(nom['result']['cases'],push['result']['cases']):
    ns,ps=n['statistics'],p['statistics']
    table+=f"| {ns['command_m_s']:g} | {ns['reset_frame_forward_speed_m_s']:.3f} | {ps['reset_frame_forward_speed_m_s']:.3f} | {ns['survival']:.0%}/{ps['survival']:.0%} | {', '.join(n['failures']) or 'Pass'} |\n"

block=f'''## October 4, 11:37 UTC adaptive check: walking qualified; take-off diagnosis

Observed {e['recorded_utc']}. R31 remains running at {state['stage']} with
{state['environment_count']:,} environments and {state['total_training_iterations']:,}
completed main updates, plus its separately recorded inherited history.
Exact saved checkpoints have now passed matching nominal/pushed/stopped/prior-speed
gates at 0.65 and 1.0 m/s. The 1.0 checkpoint is
`035_walk_100_train_006/candidate.pt`; nominal actual signed speed was
{walk['reset_frame_forward_speed_m_s']:.3f} m/s, 100% survival and 22/23 alternating
landings. These passes qualify that saved actor; the current running candidate
still fails some slow-speed retention and has no flight.

Latest matching gates: `{Path(nom['folder']).name}` and
`{Path(push['folder']).name}`, checkpoint `{Path(nom['checkpoint']).parent.name}/candidate.pt`.
The 1.5 case moves {current[1.5]['statistics']['reset_frame_forward_speed_m_s']:.3f} m/s
with 100% survival, 29/30 alternating landings, about 1.64 Hz cadence and zero
flight after {running_updates} running updates. The running stage remains unqualified. Slow tracking is nonmonotonic;
0.15 briefly recovered in gate005 but failed again in gate006. No 5 m/s or
hardware-ceiling claim is supported.

CPU-only new diagnostics reconstruct the exact 400-input actor history from
saved physical traces (maximum action discrepancy {a['actor_audit']['reconstruction']['max_abs_action_error']:.2g}).
Commands/actions match and the actor responds to command-channel interventions.
Saved 50Hz contact forces and actual sole clearances independently confirm that
one foot remains loaded in every intended running flight window. Sampled
pose-dependent CAD motor speeds and declared reductions show no sustained leg
velocity ceiling; 10Hz sampling cannot exclude brief substep clipping.
The actual runtime action clamp is +/-3; the audited running trace has 31.6%
of raw outputs outside it, concentrated in some hip/ankle channels. This is a
diagnostic lead, not grounds for enlarging physical limits.

{len(test['checks'])} new controlled reward counterexamples pass. An ideal run
beats fast walking, sliding, backward/in-place stepping and same-foot hopping
at 1.5 through 5 m/s in the fixtures. The ideal-run/fast-walk gap is only about
0.09-0.11 reward-rate units under held posture/costs. Reducing a loaded foot's
force from 338 to 18 N does not change the binary gait/flight objective while
the loaded-contact classification remains the same. A continuous phase-conditioned
force signal is an undeployed hypothesis requiring a controlled physical test.
Research and diagnostics do not prove convergence or hardware feasibility.

Preserve R31 for now: it newly qualified both walking stages since the preceding
check; standing, heading, step alternation and resources remain healthy, while
early running retention is still changing. Do not erase that actor or alter
live frozen sources solely because the timer fired. The next check must be
critical about persistent zero flight: if newer independent gates show no
flight/retention progress, perform a bounded take-off/phase intervention at an
owned checkpoint boundary, compare physical force/clearance and objective effects,
then implement a justified focused repair with fresh frozen sources and smoke
tests. Do not spend another check merely changing arbitrary weights.

Completed-main throughput: {e['completed_main_active_PPO_samples_s']:,.0f} active
PPO samples/s, {e['completed_main_worker_effective_samples_s']:,.0f} including
worker startup/exit, and {e['full_chain_completed_main_samples_s']:,.0f} over
the whole R31 chain including startup, capacity and independent gates. Last
64-update block: {last['active_PPO_samples_s']:,.0f}/{last['worker_effective_samples_s']:,.0f}.
All completed main blocks used 8192 without recovery; minimum free VRAM
{min_vram} MiB, host available {min_host} MiB. Last block private memory grew
{last['last_private_mb']-last['first_private_mb']} MiB; pre-update KL and command
mismatch stay zero. All 46 frozen hashes and one actual CUDA worker verify.
Keep target8192, reserve300, fallback/180s healthy ramp and selected iteration4.

Report and reproducible evidence:
`E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1137/REPORT.md`.
The same adaptive three-hour heartbeat remains active until qualified actual
5 m/s or a reproduced relevant velocity ceiling; then pause and brief. Leave
the computer on and preserve unsaved user work.
'''

report='# Adaptive training check — October 4, 11:37 UTC\n\n'+block+'\n## Latest candidate cases\n\n'+table+'''

## Exact checkpoint qualification

'''
for q in e['qualifications']:
    p=q['promotion']
    report+=f"- {p['stage']} ({p['speed']:g} m/s): `{p['checkpoint']}`.\n"
    if 'nominal' in q:
        report+=f"  Matching gates: `{q['nominal']}` and `{q['pushed']}`; SHA256 `{q['checkpoint_sha256']}`.\n"
report+='''

## Diagnostic files

- [Evidence snapshot and complete gate history](evidence.json), recorded at the time above.
- [CPU running/actor/CAD audit](running_audit.json) and [reproducible code](audit_running.py).
- [Running reward counterfactuals](running_counterfactuals.json) and [code](counterfactual_running.py).
- [Running contact timing](running_contact_timing.png): saved physical forces, sole clearance and knee angles, not a rendered simulation.
- [Independent speed trend](speed_tracking_trend.png).
- [Snapshot recorder](record_evidence.py). It does not start a simulator or edit a run.

## Primary-source review and decision limits

The public [G1 Isaac Lab configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
and [H1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
use alternating contact phases with duty 0.55. With half-cycle leg offsets this
implies overlapping stance rather than flight. That mathematical inference is
why this project's 0.43 running duty and flight term are declared local adaptations;
the public walking configuration alone does not validate 5 m/s running here.

[Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2)
build gait objectives around phase-dependent physical forces and foot velocities,
with smooth phase boundaries. This supports testing a continuous unloading signal
if the measured take-off plateau persists. It does not establish that the proposed
local exponential signal, its scale, this robot's transmissions or this actor will
work. The force-sweep fixture holds posture, velocity and other costs fixed; it
is neither a dynamically possible trajectory nor a substitute for a physical test.

This check changes diagnostics and documentation only. Runtime reward, PPO,
action mapping, geometry, collisions, mass/inertia, motor limits and gates are
unchanged. No new frozen run, policy reset, qualification relaxation or shutdown.
The current actor can continue learning; its preserved qualified walking actor
is available if a later tested repair needs a known good starting point.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')

marker='adaptive_check_20261004_1137'
wrapped=f'<!-- {marker} -->\n'+block+f'<!-- end_{marker} -->\n\n'
paths=[HERE/'automation/THREE_HOUR_CHECK.md',
       HERE.parents[1]/'docs/robot/TRAINING_JOURNAL_20261001.md',
       Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
       Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
       Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
for path in paths:
    text=path.read_text(encoding='utf-8-sig')
    text=re.sub(r'<!-- '+marker+r' -->.*?<!-- end_'+marker+r' -->\s*','',text,flags=re.S)
    # Workflow constraints stay first; dated entries stay below them.
    if path.name=='THREE_HOUR_CHECK.md':text=text.rstrip()+'\n\n'+wrapped
    else:text=wrapped+text
    path.write_text(text,encoding='utf-8')
history=HERE/'ITERATION_HISTORY.md'
text=history.read_text(encoding='utf-8-sig')
text=re.sub(r'<!-- '+marker+r' -->.*?<!-- end_'+marker+r' -->\s*','',text,flags=re.S)
text=re.sub(r'\*\*Current recorded state:\*\*.*?(?=\n\| Read next)',
            '**Current recorded state:** R31 has saved qualified standing and walking\n'
            'checkpoints through 1.0 m/s; its current 1.5 m/s candidate remains\n'
            'unqualified due to zero flight and slow-speed tracking failures. The\n'
            'October 4 11:37 UTC check added CPU command/history, physical force/sole,\n'
            'CAD motor speed and reward feedback diagnostics; it preserves the live\n'
            'improving chain and its exact qualified walking actor. Read fresh state\n'
            'and the newest dated report. The three-hour checks remain active toward\n'
            'actual qualified 5 m/s or a proven relevant velocity ceiling, followed\n'
            'by pause and briefing; the computer stays on. Older statuses are dated.\n',text,flags=re.S)
anchor='## Physical-v29 runs, October 1–4'
text=text.replace(anchor,wrapped+anchor,1)
history.write_text(text,encoding='utf-8')

focus=f'''Latest dated focus, October 4, 11:37 UTC check: R31_exact_command_rehearsal remains active at run_150, {state['total_training_iterations']} completed main updates, 8192 environments. It newly qualified0.65 at checkpoint015 (matching gates016/017) and1.0 at checkpoint035 (matching gates036/037): nominal actual1.0-command signed speed {walk['reset_frame_forward_speed_m_s']:.3f}m/s,100% survival,22/23 alternating landings, stopped and all prior commands pass. These qualifications belong to the exact saved actors; the current running candidate is unqualified. Read diagnostics/adaptive_check_20261004_1137/REPORT.md, evidence.json, running_audit.json and running_counterfactuals.json. Latest matching gates{Path(nom['folder']).name}/{Path(push['folder']).name} at checkpoint{Path(nom['checkpoint']).parent.name} show1.5 command achieves{current[1.5]['statistics']['reset_frame_forward_speed_m_s']:.3f}m/s nominal,100% survival,29/30 alternating landings,1.64Hz cadence and zero flight. Retained0.15/0.35 again fail speed; slow tracking is nonmonotonic, not solved. The1.5 stage has completed320 updates with no independent flight. A saved50Hz physical trace confirms one loaded foot in every intended flight window, no intervals with both forces below1N or both soles above1mm. Reconstructed400-input histories reproduce2240 saved actor samples within4.8e-6 maximum action error; observed/action commands match and fixed-state command interventions affect leg outputs. Runtime action clamp is +/-3;31.6% of raw running outputs exceed it. Investigate effective action/target gradients and timing, not just raw derivatives. All-population10Hz motor speeds use exact CAD Jacobians/reductions; no sustained leg ceiling, although isolated ankle speeds approach95% of cap and substep clipping is not excluded. Eleven frozen-reward counterexamples pass, including ideal run versus fast walk at1.5 through5. The binary gait/flight objective is flat while a loaded foot is reduced from338 to18N; a continuous phase-conditioned force signal is an undeployed research hypothesis, not proven dynamics. Preserve useful walking/standing actors and the current chain while fresh independent retention is changing; no runtime source or reward changed this check. Next compare newer flight, slow retention, action clipping and objective trends against this baseline. If flight remains zero and retention does not improve over the newer blocks, do not merely repeat reward-identical waiting: at an owned saved boundary perform a bounded physical take-off/phase intervention and per-command objective/actor-response test, then implement a justified focused repair with CPU counterexamples, physical/PPO/load smoke and fresh frozen sources. Primary sources reviewed are public Unitree G1/H1 velocity tasks and Siekmann et al. Periodic Reward Composition (arxiv2011.01387); public0.55 walking duty alone is no running guarantee. Preserve iteration4, geometry, CAD/motor limits and all gates. Completed-main means about{e['completed_main_active_PPO_samples_s']/1000:.1f}k active PPO/{e['completed_main_worker_effective_samples_s']/1000:.1f}k worker-effective/{e['full_chain_completed_main_samples_s']/1000:.1f}k full-chain samples/s; denominators differ. Minimum completed-main VRAM{min_vram}MiB and host{min_host}MiB, stable private/CUDA memory; zero pre-update KL and action-command mismatch,46 hashes verify and one actual CUDA worker. All completed main blocks used8192 without recovery, target8192/reserve300/fallback/180s healthy ramp remain. Read fresh ACTIVE_RUN/state instead of assuming this dated run persists. Keep this same adaptive three-hour heartbeat active; no5m/s or velocity-ceiling endpoint is proven. Leave the computer on.'''
focus=focus.replace('320',str(running_updates))
(OUT/'automation_focus.txt').write_text(focus,encoding='utf-8')

next_check=dict(recorded_utc=e['recorded_utc'],active_run=str(run),status=state['status'],stage=state['stage'],
                checkpoint=state['checkpoint'],checkpoint_sha256=e['current_checkpoint_sha256'],
                source_hashes={k:v['actual'] for k,v in e['frozen_hashes'].items()},
                independently_qualified_speeds=[p['speed'] for p in state['promotions']],
                exact_qualification=e['qualifications'],candidate_qualified=False,
                current_curriculum_max_m_s=5.,main_updates=state['total_training_iterations'],
                prior_main_updates=state.get('prior_training_iterations'),
                latest_nominal=nom,latest_pushed=push,completed_main_throughput=dict(
                    active_PPO_samples_s=e['completed_main_active_PPO_samples_s'],
                    worker_effective_samples_s=e['completed_main_worker_effective_samples_s'],
                    full_chain_samples_s=e['full_chain_completed_main_samples_s'],
                    scope='Completed main PPO samples; full-chain includes startup/capacity/eval, worker-effective includes startup/exit, active excludes them'),
                last_completed_main=last,current_worker_progress=e['current_worker_progress'],
                motor_and_actor_audit=str(OUT/'running_audit.json'),counterfactuals=str(OUT/'running_counterfactuals.json'),
                action_taken='Preserved newly successful walker/current early-running actor; added tested CPU history/command/physical-force/clearance/motor/reward diagnostics, no live-source change',
                next_focus=[
                    'Compare newer independent flight and stopped/prior-speed retention against320 running updates; do not treat fast walking as running or old trusted passes as current candidate qualification.',
                    'If zero-flight and retention plateau persist, perform bounded physical take-off/phase and effective action-gradient intervention at an owned saved boundary; test a justified phase-force repair in a fresh frozen continuation while preserving good walking actors.',
                    'Continue8192 memory/actual-throughput checks and CAD motor-speed quantiles; a ceiling needs controlled substep clipping/plateau evidence and current qualification.'],
                report=str(OUT/'REPORT.md'),automation_focus=focus,
                endpoint='Qualified actual5m/s or proven locomotion-relevant velocity ceiling; save/pause training and same automation/brief; computer stays on')
next_check['next_focus']=[f.replace('320',str(running_updates)) for f in next_check['next_focus']]
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n',encoding='utf-8')
(OUT/'documentation_update.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                    files=[str(p) for p in paths]+[str(history),str(HERE/'automation/NEXT_CHECK.json')],
                    runtime_changed=False,qualification_relaxed=False,report=str(OUT/'REPORT.md')),indent=2),encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),latest_checkpoint=state['checkpoint'],qualified_speeds=next_check['independently_qualified_speeds'],
                     document_files=len(paths)+2),indent=2))
