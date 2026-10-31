"""Update local training indexes; preserve historical text and never invoke Git."""
from pathlib import Path
import json, shutil, re

HERE=Path(__file__).parent
REPO=Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
DOCS=REPO/'docs/robot'
SKILL=Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')
DIAG=HERE/'diagnostics/unitree_port_20261002'
BACKUP=DIAG/'documentation_before_port'
MARK='<!-- unitree-port-20261002 -->'

current='''The current replacement is **r20_unitree_g1**. Read
`E:/Aditya/robot_sim/physical_v29/training/ACTIVE_RUN.json`, its live `state.json`,
verified worker/supervisor command lines, fresh progress and independent gates.
R19 was drained and preserved without qualifying walking. R20 starts a fresh
five-frame, nominal-position-offset policy using pinned Unitree RL Lab G1
reward bodies/weights and PPO settings. Old CAD-IK moving action references,
discovery bonuses and the standing teacher are absent. H1 RL Lab, older RL Gym
H1/G1 and public MJLab H2 tasks were compared separately, not blended.

The reward version is `unitree_g1_port_v1`; action/observation contract is
`physical_v29_history5_unitree_defaultoffset_v1`. Actor/critic dimensions are
400/415. Original G1 walking phase parameters remain; requested running uses
an explicitly documented cadence/flight extension. Robot dimensions and
entity mappings are adapted; saved physical v29 mass, axes, actuator limits,
collisions and CAD are unchanged. CAD v32 remains a separate release.

5,589 CPU assertions, actual physics/PPO/load checks and fresh capacity
benchmarks passed. At 8192 the short benchmark measured 11,674 env steps/s and
1533 MiB minimum sampled free VRAM. 280 real scripted probe screenshots, all
47 official reference GIF frames, and 50 learned-smoke CPU reconstruction
frames were reviewed. The four-update candidate failed standing/walking;
scripted motions are not a learned gait. No walking/style success is claimed.

Target 8192, reserve 300 MiB, automatic fallback and 180-second healthy-live-PPO
ramp remain. RAM recycling preserves checkpoints/Adam/LR, but the underlying
allocation leak is unresolved. Stages: stand, walk 0.35/0.65/1.0, run
1.5/2.0/2.5 m/s; 0.15 remains a retention command. Promotion requires current
and prior nominal/pushed physical gates, never reward or smoke alone. No camera
is instantiated in training and no second Isaac/CUDA worker may overlap it.
Do not load old history-4/IK or another robot's pretrained tensors into R20.
'''

def preserve(path):
    BACKUP.mkdir(parents=True,exist_ok=True)
    key=str(path).replace(':','').replace('\\','_').replace('/','_')
    dest=BACKUP/(key+'.txt')
    if not dest.exists():
        shutil.copy2(path,dest)

def prepend(path,title,text):
    old=path.read_text(encoding='utf-8')
    if MARK in old:
        return
    preserve(path)
    path.write_text(title+'\n\n'+MARK+'\n\n'+text+'\n\n## Historical record before the Unitree port\n\n'+old,encoding='utf-8')

def main():
    active=json.loads((HERE/'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
    assert Path(active['run']).name=='locomotion_20261002_r20_unitree_g1'
    prepend(HERE/'README.md','# Physical-v29 Unitree reward training',current+'''
Details, original source pins, reward comparisons, tests and scope:
[Unitree port report](UNITREE_REWARD_PORT_20261002.md).
[Frame/reward review](diagnostics/unitree_port_20261002/frame-review.html).
Use `monitor_training.ps1 -Watch` to inspect the running chain. The older
launch/play commands below belong to historical contracts; do not use them
for this active five-frame policy.
''')
    prepend(HERE/'RESEARCH.md','# Unitree source comparison and reward port',current+'''
The latest primary-source comparison and adaptations are in
[UNITREE_REWARD_PORT_20261002.md](UNITREE_REWARD_PORT_20261002.md).
Pinned copied training source/hashes: `unitree_vendor/sources.json`.
H2/RL Gym source and official GIF hashes:
`diagnostics/unitree_port_20261002/reference/sources.json`.
The older research-guided residual method below is history; it is no longer
the active action/reward system.
''')
    prepend(SKILL,'# Current physical-v29 Unitree port — October 2, 2026',current+'''
Read training/UNITREE_REWARD_PORT_20261002.md and its evidence. H1's positive
clearance weight 20 can obscure failure in a total score; this property was
tested, not presumed to be the cause of old v29 failures. G1 still gives
clearance credit to stationary feet and gait credit to in-place alternation.
Static nominal COM inside the combined sole AABB does not establish dynamic
support. Public reward copying does not guarantee Unitree commercial style.
''')
    file_locations=REPO/'Learning/HumanoidTraining/FILE_LOCATIONS.md'
    old=file_locations.read_text(encoding='utf-8')
    if MARK not in old:
        preserve(file_locations)
        insertion='\n\n'+MARK+'\n\n## Current training: Unitree G1 port — October 2, 2026\n\n'+current+'''
Port report: `E:/Aditya/robot_sim/physical_v29/training/UNITREE_REWARD_PORT_20261002.md`.
Interactive evidence: `training/diagnostics/unitree_port_20261002/frame-review.html`.
Older r19/r17/r15 training entries below are preserved historical evidence;
their dated active pointers do not describe the current run.
'''
        head,rest=old.split('\n',1)
        file_locations.write_text(head+insertion+'\n'+rest,encoding='utf-8')
    training=DOCS/'TRAINING.md'
    preserve(training)
    training.write_text('# Physical-v29 Unitree G1 training — October 2, 2026\n\n'+MARK+'\n\n'+current+'''
Full report: [Unitree rewards, tests and visual review](TRAINING_UNITREE_PORT_20261002.md).
Authoritative evidence and live status are in the workspace training folder.
The dated report is not proof of process liveness or completion. Local only;
no Git commit/push. Old training documents are preserved in the workspace's
diagnostic documentation backup and existing repository history directories.
''',encoding='utf-8')
    report=(HERE/'UNITREE_REWARD_PORT_20261002.md').read_text(encoding='utf-8')
    def absolute_link(match):
        target=match.group(1)
        if target.startswith(('https://','http://','#')):
            return match.group(0)
        path=HERE/target
        return ']('+path.resolve().as_posix()+')' if path.exists() else match.group(0)
    report=re.sub(r'\]\(([^)]+)\)',absolute_link,report)
    (DOCS/'TRAINING_UNITREE_PORT_20261002.md').write_text(report,encoding='utf-8')
    journal=DOCS/'TRAINING_JOURNAL_20261001.md'
    old=journal.read_text(encoding='utf-8')
    if MARK not in old:
        preserve(journal)
        journal.write_text(old+'\n\n## October 2: pinned Unitree reward replacement\n\n'+MARK+'\n\n'+current+'''
See TRAINING_UNITREE_PORT_20261002.md for per-term source mappings and limits.
The local frame review has actual scripted screenshots with reward traces and
separately labeled learned-smoke reconstructions. Falling was not called
walking. Render-process retries were preserved; successful D3D12 recording
completed before the new chain launched. No training camera or extra GPU
worker is left running by the visual review.
''',encoding='utf-8')
    print('Updated local README/research, skill reference, file index and training journal; backups preserved; no Git commands.')

if __name__=='__main__':
    main()
