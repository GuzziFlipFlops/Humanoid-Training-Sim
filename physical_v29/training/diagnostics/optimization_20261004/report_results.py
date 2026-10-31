"""Build an evidence-backed local report after optimization and restart checks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
def read(path):return json.loads((here/path).read_text())
paths={'baseline':'baseline/benchmark_4096','01':'iteration_01/benchmark_4096',
       '02':'iteration_02/benchmark_4096','04':'iteration_04/benchmark_4096_retry','05':'iteration_05/benchmark_4096'}
bench={key:read(value+'/result.json') for key,value in paths.items()}
baseline=bench['baseline']['samples_per_second']
descriptions={
 '01':'Prepared immutable geometry and shared sine/cosine values; original arithmetic ordering retained.',
 '02':'Bounded owned CUDA graphs for each exact differential state calculation; fresh pose copied every call.',
 '03':'First attempt to graph the entire PD and coupled hardware actuator calculation.',
 '04':'Fixed capture-unsafe Python joint-index conversion with prepared CUDA index tensors; full actuation graph.',
 '05':'Added pure graph calls for target projection and fresh post-physics differential validity; two declared environment-loop replacements.',
}
rows=[]
for key in ['01','02','03','04','05']:
    if key=='03':
        rows.append('| 3 | Full actuator capture, initial attempt | Failed capture | Rejected: CPU-to-GPU index conversion during capture. Logs/source retained. |')
    else:
        value=bench[key]['samples_per_second']
        decision={'01':'Valid, superseded','02':'Valid, superseded','04':'Selected','05':'Valid, rejected: +1.5% is within short-run variation; additional environment changes'}[key]
        rows.append(f'| {int(key)} | {descriptions[key]} | {value:,.0f} ({value/baseline:.2f}× baseline) | {decision} |')
ppo={count:read(f'iteration_04/ppo_{count}/result.json') for count in [4096,6144]}
old=read('baseline/ppo_4096/result.json')
parity=read('iteration_04/mixed_replay_parity.json')
integration=read('continuation_integration.json')
assert parity['all_bitwise_equal'] and integration['passed']
active_path=here.parents[1]/'ACTIVE_RUN.json'
active=json.loads(active_path.read_text())
run=Path(active['run']);state=json.loads((run/'state.json').read_text())
records=dict(updated_utc=datetime.now(timezone.utc).isoformat(),selected='04',controlled_inference=bench,
             ppo_baseline=old,ppo_selected=ppo,source_audit=read('source_audit.json'),
             physical_replay_parity=parity,integration=integration,
             active_run=str(run),active_status=state['status'],active_operation=state.get('operation'),
             semantics='Identical execution outputs are not locomotion or real-hardware qualification')
capacity=json.loads((run/'benchmarks.json').read_text()) if (run/'benchmarks.json').exists() else []
records['fresh_production_capacity']=capacity
(here/'results.json').write_text(json.dumps(records,indent=2))
text=f'''# Five model-execution optimization iterations — October 4, 2026

Selected and applied **iteration 4**. The controlled 4,096-environment inference
benchmark rose from **{baseline:,.0f} to {bench['04']['samples_per_second']:,.0f} samples/s
({bench['04']['samples_per_second']/baseline:.2f}×)**. Actual same-count PPO rose from
**{old['active_ppo_samples_per_second']:,.0f} to {ppo[4096]['active_ppo_samples_per_second']:,.0f}
samples/s ({ppo[4096]['active_ppo_samples_per_second']/old['active_ppo_samples_per_second']:.2f}×)**.
At 6,144 environments it measured **{ppo[6144]['active_ppo_samples_per_second']:,.0f} PPO samples/s**.
This speeds data collection; it does not prove the current walker or guarantee
convergence. The current 0.35 m/s chain still requires matching independent gates.

## All iterations, including the rejected attempt

The user requested at least three and at most five iterations. Exactly five
source variants were created and tested. The unchanged baseline is a reference,
not a sixth optimization. Repeated tests and integration checks did not create
additional model variants.

| Iteration | Actual implementation | Controlled inference samples/s | Decision |
| --- | --- | --- | --- |
| Baseline | R29 frozen model and full 256-update checkpoint | {baseline:,.0f} | Reference |
{chr(10).join(rows)}

Each directory has a frozen `source/physical_v29`, SHA manifest, launch arguments,
logs, results and trace arrays. The iteration-3 capture failure and iteration-4
guarded startup retry remain preserved. Failed CUDA capture and reserve-breach
workers hung in Kit close; only their verified owned processes were drained.
The one-worker guard blocked the next launch until they were drained. No
unrelated process or unsaved Fusion document was closed.

## Fidelity and why this is not an excessive simplification

The training asset already has 24 box colliders/rigid links and 23 revolute
joints, with no stereo vision, finger dynamics or passive closed-loop solver.
The speedup executes the same pose-dependent rod differential math more
efficiently. No physical CAD part or motor transform was moved or removed.

Iteration 4 changes only `cad_differential.py`, `hardware_actuators.py`, and the
new `owned_cuda_graph.py`. All other experiment runtime files are byte-identical
to the frozen baseline. Preserved: joint axes/limits, nominal offsets, collision
shapes/offsets, masses/inertias (34.46334 kg estimate), 95% assumed transmission
efficiency, motor torque and speed limits, simultaneous pitch/roll envelopes,
overspeed braking, gravity, self-collision, contact sensors, 5 ms physics steps,
four substeps, 8/2 solver iterations, actions, observations/history ownership,
resets/randomization, commands, rewards and independent gates.

Immutable geometry is prepared once per device/dtype. Each graph copies the
current inputs into static storage; no Jacobian is reused across a changed
pose. Results are cloned, so another replay cannot overwrite saved rollout
data. Caches have at most two graph entries per operation. CPU/gradient-bearing
inputs use eager tensor calculations.

Verification evidence:

- [Source audit](source_audit.json): every experiment manifest matches; every
  undeclared file is identical. R29's checkpoint was unchanged.
- [CPU math](iteration_01/math_cpu.json) and [CUDA graph math](iteration_02/math_cuda.json):
  bitwise equality for 13,513 wide-grid/random/zero poses for each of three
  differentials and both float32/float64, five coupled torque/speed/braking
  cases, changed shapes, noncontiguous inputs and owned prior outputs. Iteration
  4 has the exact same differential-file SHA as the CUDA graph test.
- [Full actuator parity](iteration_04/benchmark_4096_retry/actuator_parity.json):
  all eight optional position/velocity/feedforward modes, both dtypes, changed
  gains/poses/velocities and previous-output ownership pass bitwise comparison.
- [4,096-robot sampled replay](iteration_04/physical_parity_4096.json): all saved
  observations, joints, torques, root states, actions, targets, commands, rewards,
  termination flags, contacts, foot forces/positions and COM are bitwise equal.
- [Every-frame 72-robot replay](iteration_04/mixed_replay_parity.json): all 400
  frames and all 72 robots, 0/0.15/0.35/0.65/1/1.5 m/s commands, forced resets at
  tick 100 and lateral velocity pushes at tick 150, remain bitwise equal in
  every saved field. High commands in this diagnostic are not gait qualification.
- Actual eight-update PPO runs at 4,096 and 6,144 retain the full actor, critic,
  exploration, Adam, adaptive LR and training-only standing teacher. Every
  sampled pre-update KL is zero; checkpoint save and full reload succeed.
- [Continuation integration](continuation_integration.json): exact three-file
  allowlist and pinned reports; ten source-mutation counterexamples and a false
  locomotion-qualification claim are rejected. The real freeze path includes
  and hashes the new graph helper. Original checkpoints/frozen runs are intact.

## Throughput denominator and resources

Inference comparisons used seed 901, the same R29 actor, 4,096 environments,
24 warmup ticks and three 64-tick measured windows in fresh processes. The
observation returned by the previous step is consumed, matching PPO. The
earlier 24-tick profiler requested an extra observation each tick and added
instrumentation; its 7.4k figure is not this controlled baseline.

| Actual PPO test | Active samples/s | Effective samples/s | Scope |
| --- | --- | --- | --- |
| Baseline 4,096 | {old['active_ppo_samples_per_second']:,.0f} | {old['effective_ppo_samples_per_second']:,.0f} | Eight updates, full continuation |
| Iteration 4, 4,096 | {ppo[4096]['active_ppo_samples_per_second']:,.0f} | {ppo[4096]['effective_ppo_samples_per_second']:,.0f} | Eight updates, full continuation |
| Iteration 4, 6,144 | {ppo[6144]['active_ppo_samples_per_second']:,.0f} | {ppo[6144]['effective_ppo_samples_per_second']:,.0f} | Eight updates, full continuation |
| Iteration 4, 8,192 | Not eligible | Not eligible | VRAM dropped to 214 MiB; reserve check stopped test |

Active PPO counts collection plus optimizer time. Effective PPO also counts
in-process logging/GC/guards, but excludes Isaac process initialization and
independent qualification. The whole-chain rate will be lower and must be
measured separately. These short tests do not certify long-term memory stability.
The archived old-model rate and public G1 RTX4090/Linux benchmarks are different
models/hardware. No comparable throughput guarantee is made.

## Selected continuation and preservation

Original R29 checkpoint SHA:
`12b02dd709175b1e2ddf1f3cc6a6f4334f943bc21bd423879acf92d18b8031b1`.
All benchmark-updated actors are diagnostic and discarded for main learning.
The exact original full checkpoint and qualified standing teacher seed the
fresh source-frozen continuation; the critic and Adam are not reset for an
execution-only change. The supervisor performs a fresh physical preflight,
PPO/load smoke, complete capacity tests and current nominal/pushed/stopped/
earlier-speed gates. Execution parity does not qualify walking or running.

Continuation at report time: `{run.name}`, status `{state['status']}`,
operation `{state.get('operation','not recorded')}`. The authoritative fresh
state is `{run}/state.json`. Target remains 8,192 with a 300 MiB VRAM reserve,
host guard, fallback and 180 continuously healthy seconds before an upward ramp.
Four-hour prompts remain disabled. No reward redesign, CAD geometry change,
commit/push, forced unsaved-document close or shutdown occurred.

The CUDA graph design follows [PyTorch's primary CUDA graph guidance](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/):
static storage, warmup and replay restrictions. See [experiment README](README.md),
[CPU/hardware helpers](actuator_parity.py), [reproducible benchmark](benchmark.py),
and [pinned continuation proof](continuation_proof.json).
'''
if capacity:
    text+='\n## Fresh production capacity after restart\n\n'
    text+='These use the actual frozen worker at stopped command, without the walking\nstanding-teacher gradient. They are distinct from the walking diagnostic above.\n\n'
    text+='| Environments | Completed updates | Active samples/s | Minimum free VRAM MiB | Eligible |\n| --- | --- | --- | --- | --- |\n'
    for row in capacity:
        text+=f"| {row['environments']:,} | {row['updates']} | {row['environment_steps_per_second']:,.0f} | {row['minimum_free_mb']} | {row['eligible']} |\n"
    text+=f"\nCurrent stage: `{state.get('stage')}`; operation `{state.get('operation')}`. "
    text+=f"The selected next training count is {state.get('next_training_environment_count','not selected')} environments. "
    text+='The 8,192 stopped-command test completed 24 updates above the reserves; walking must still obey the runtime guard and may fall back. No current walking qualification is inferred from capacity.\n'
(here/'REPORT.md').write_text(text)
print(json.dumps(dict(report=str(here/'REPORT.md'),selected='04',active_run=str(run),status=state['status'])))
