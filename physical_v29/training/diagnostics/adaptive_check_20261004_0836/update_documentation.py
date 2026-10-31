"""Update current references and preserve the dated iteration record locally."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'physical_v29/training'
OUT=Path(__file__).parent
data=json.loads((OUT/'evidence.json').read_text());state=data['live_state']
marker='<!-- adaptive_check_20261004_0836 -->'
end='<!-- end_adaptive_check_20261004_0836 -->'
def upsert(path,body,prepend=False):
    path=Path(path);text=path.read_text(encoding='utf-8-sig')
    section=marker+'\n'+body.strip()+'\n'+end+'\n'
    if marker in text:
        a=text.index(marker);b=text.index(end,a)+len(end)
        text=text[:a]+section.rstrip()+text[b:]
    elif prepend:text=section+'\n'+text
    else:text=text.rstrip()+'\n\n'+section
    path.write_text(text,encoding='utf-8')

short=f'''## October 4, 08:36 UTC adaptive check: R31 retained-command rehearsal

Current continuation: `locomotion_20261004_r31_exact_command_rehearsal`,
{state['status']}, {state['stage']}. Read fresh ACTIVE_RUN.json/state.
R30 independently qualified0.35, improved straight alternating0.65, but late
0.15 retention repeatedly oversped at0.27–0.30. It is preserved at1,472 main
updates, full checkpoint078, SHA8a62733db47184082e9a1dcf5459893c9285f10018feb5af3a0b3f86f202e31d.
R31 retains that full actor/critic/Adam/exploration and standing teacher. The
existing restart learning-rate cap remains1e-4. Training-only exact prior/frontier
command rehearsal gives0.15 about6.71 times more exposure at the0.65 stage;
all reward weights, inference/history/action/reset behavior, physical assets,
CAD Jacobians and motor limits remain. Iteration4 execution stays selected.

Current0.65,1.0 and running qualification is incomplete; prior0.35 remains tied
to its original exact trusted checkpoint. No threshold is relaxed. The frozen
speed list now extends through3/3.5/4/4.5/5 m/s with32 trials per command and an
additional measured-forward-speed requirement at5. The entire128-robot,
20-second same-actor nominal baseline and all saved synchronized arrays match
R30 bit for bit. CPU command/provenance/running counterfactuals and real CUDA
physics/PPO/full-checkpoint-load checks passed; these are not learned qualification.

Latest R30 main6144 throughput was about35.7k active PPO,28.7k worker-effective
and15.3k full-chain samples/s, with stable memory. R31 stopped-command capacity
completed at8192:41.6k active PPO,659 MiB minimum GPU free and9439 MiB minimum
host available. Moving training still must obey reserve300, host guards,
fallback and180 continuously healthy seconds for an upward ramp. Preserve unsaved
Fusion work. One actual CUDA worker; no stereo.

Report/proof/tests: `E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_0836/REPORT.md`.
Complete local chronology: `physical_v29/training/ITERATION_HISTORY.md`.
Next focused tests: slow/current-speed retention, moving memory/throughput,
then running gait and motor velocity diagnostics. Keep improving actors.
The same adaptive heartbeat remains active every3 hours, GPT-6.1 Sol/max;
actual5 m/s qualification or a reproduced relevant velocity ceiling ends it
with evidence saved, training/checks paused and a briefing. Computer stays on.
'''
if data.get('last_completed_main'):
    block=data['last_completed_main']
    short+=f"\nR31's latest completed main block has {block['iterations']} updates at{block['environments']} environments, " \
        f"{block['active_PPO_samples_s']:,.0f} active PPO and{block['worker_effective_samples_s']:,.0f} worker-effective samples/s. " \
        f"Minimum free VRAM{block['minimum_free_vram_mb']}MiB; host available{block['minimum_host_available_mb']}MiB. " \
        f"Pre-update KL and action/command mismatch remain zero.\n"
post=next((g for g in reversed(data.get('current_gates',[])) if g['folder'].endswith('_nominal') and 'gate_001_' not in g['folder']),None)
if post:
    speeds={c['statistics']['command_m_s']:c['statistics']['reset_frame_forward_speed_m_s'] for c in post['cases']}
    short+=f"\nLatest post-rehearsal nominal test ({post['folder']}): retained0.15 now{speeds[.15]:.3f}m/s versus0.277 baseline; " \
        f"current0.65 now{speeds[.65]:.3f}m/s versus0.732. Survival remains100%. " \
        "Slow tracking still fails: nominal speed fell from0.277 to0.258 at64 updates, then rose back to0.268 at128. " \
        "Current-speed tracking continues improving. Keep the actor and require matching pushed/retention gates; the slow-control issue is not solved.\n"
history=HERE/'ITERATION_HISTORY.md'
text=history.read_text()
a=text.index('**Current recorded state:**');b=text.index('| Read next |',a)
text=text[:a]+'''**Current recorded state:** R31 exact-command rehearsal is the fresh full-state
continuation of R30's1,472-main-update actor. R30 independently qualified0.35 m/s;
straight0.65 cases improve/pass while retained0.15 overspeed blocks promotion.
Training-only rehearsal changes exposure, retaining all current rewards and
physical/optimizer state. The current frozen curriculum extends to5 m/s.
Fresh CPU/CUDA/load tests and a bitwise identical independent baseline are saved.
Iteration4 execution remains selected; target8192 retains reserve300/fallback/
180-second host/GPU ramp. Read the latest dated report and fresh state.
The same three-hour adaptive checks end only at qualified actual5 m/s or a
reproduced relevant velocity ceiling, followed by pause and briefing; computer
stays on. Older dated statuses below remain historical evidence.

'''+text[b:]
history.write_text(text)
upsert(history,short+'''
Preparation failures are retained in the dated report: exact-newline restoration,
test fixture dtype/source alignment, and harmless drain-monitor folder lookup.
No production source was edited in place and no checkpoint was discarded.
Historical R30 partial35-update recovery remains in its original file and is
counted in measured samples, separate from its main-update state count.
''')
row='| [Adaptive check and R31](diagnostics/adaptive_check_20261004_0836/REPORT.md) | Independent R30 progress/retention, tested exact-command rehearsal, full-state continuation and5 m/s extension. |\n'
text=history.read_text()
if row not in text:text=text.replace('## How to interpret the records',row+'\n## How to interpret the records',1)
history.write_text(text)
repo=Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
for path in [ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',repo/'docs/robot/TRAINING_JOURNAL_20261001.md']:
    upsert(path,short)
upsert(repo/'Learning/HumanoidTraining/FILE_LOCATIONS.md',short,prepend=True)
upsert(Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),short,prepend=True)
workflow=HERE/'automation/THREE_HOUR_CHECK.md'
text=workflow.read_text()
old='''existing R30 curriculum stops at 2.5 m/s: this is an intermediate milestone,
not completion of the new goal. At a suitable saved boundary, implement and test
an explicit extension through 3, 3.5, 4, 4.5 and 5 m/s, adjusting intermediate
spacing only when justified.'''
new='''R31 curriculum now extends through 3, 3.5, 4, 4.5 and 5 m/s, after the
earlier 2.5 m/s milestone. Its extension and prior-command classifiers were
counterfactually tested; at least32 trials per command remain. The additional
5 m/s gate requires measured signed forward speed within1% of the goal.
Adjust intermediate spacing only when justified by measured behavior.'''
assert old in text or new in text
text=text.replace(old,new)
workflow.write_text(text)
upsert(workflow,short)
print(json.dumps(dict(updated_history=str(history),current_run=state['stage'],local_only=True)))
