from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
brief='''<!-- adaptive_check_20261005_0841_current -->
## Latest continuation: R36, October 5

Read E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261005_0841/REPORT.md and fresh ACTIVE_RUN/state. R36 continues the exact R35 checkpoint 213 actor, critic, Adam and exploration after 4,352 new and 1,473 inherited main updates. The correction addresses reproduced dependence of the older walking teacher on tilted-head feedback. Only its training targets at 0.15, 0.35, 0.65 and 1.0 m/s are conditioned. Student observations and inference remain truthful; the head/arm loss, rewards, gates, physics, CAD motor caps, PPO and iteration 4 execution remain.

The all-prior-only correction passed stopped and every prior walking case in a matched 4,096-environment, 64-update comparison and full 192-robot, 20-second nominal/pushed tests. Head RMS improved to about 3.1–3.5 degrees. Running flight remained 0%/0.0174%, so running is unqualified. Running-only row transfer lost its initial flight gain by 128 updates and was rejected. No diagnostic learned actor was deployed. Twenty-seven CPU checks, twelve migration checks, real PPO/full-state load and bitwise parity for all 39 replay arrays pass. Three runtime files change; 44 protected files remain identical and 49 frozen hashes verify.

At the first deployed observation, R36 had 21 genuine main updates at 8,192 environments, exact checkpoint load and zero pre-update KL/command mismatch. Read new_main_evidence.json and newer gates for actual progress. The 3,731.7-second diagnostic pause is documented and excluded from training throughput. Next check prior retention stability, then current-state flight/objective diagnosis. Preserve the 8,192 target, 300 MiB reserve, 180-second healthy-memory ramp, same adaptive three-hour heartbeat and computer-on endpoint. Qualification must bind the actual saved actor and matching independent gates.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
 text=p.read_text(encoding='utf-8-sig');marker='<!-- adaptive_check_20261005_0540_current -->';assert text.startswith('<!-- adaptive_check_20261005_0841_current -->') and marker in text
 p.write_text(brief+text[text.index(marker):],encoding='utf-8')
entry=f'''<!-- adaptive_check_20261005_0841 -->
## October 5, 04:41 EDT — R36 walking teacher-target correction

[Dated report]({(OUT/'REPORT.md').as_posix()}). R35 is preserved at 4,352 new main updates plus 1,473 inherited updates, checkpoint 213 SHA ec62b406f83b5a6f78a9b4979ccedfe4200b41bd1abc7c4d8cb35037ccd71c91. Normal-limit physical cohorts reproduce dependence of the older walking teacher on head feedback. Teacher-only phase context restores gait with a physically neutral head. Diagnostic head overrides and rewritten inputs were never deployed.

Three learning variants are preserved. Running-row imitation gained 0.490%/0.576% flight after 64 updates but lost it by 128 and was rejected. Partial-prior plus running transfer oversped at 0.15 and was rejected. The selected all-four walking-prior correction passed stopped and every prior case through 1.0 in a matched 4,096-environment, 64-update comparison and full 192-robot, 20-second nominal/pushed tests. Head RMS improved to about 3.1–3.5 degrees. Running flight remained 0%/0.0174%, so running is unqualified. Only teacher training labels change; the student receives truthful observations. The original 0.25 MSE formula, 23-output denominator, stopped teacher and head/arm loss remain. Numerical prior targets and gradients intentionally change.

Twenty-seven CPU counterexamples, twelve strict migration checks, real 64-environment PPO/full-state reload and whole-replay bitwise parity for 39 arrays pass. Three runtime files change; 44 protected files remain identical and 49 hashes verify. R36 continues the original 213 full actor, critic, Adam and exploration under the existing restart learning-rate ceiling; no diagnostic actor was deployed. The owned old supervisor was stopped only after a fresh source freeze. The 3,731.7-second diagnostic pause is documented. The first deployed observation has 21 genuine updates at 8,192, exact full-state load and zero KL/command mismatch. Minimum free GPU memory was 551 MiB and host available was 6,408 MiB. Later evidence belongs in the dated report and handoff.

R35 throughput was 43.53k active collection-plus-PPO, 37.49k worker-effective and 24.48k whole-chain samples/s excluding pauses; these denominators differ. New partial throughput is not a completed-block or whole-chain rate. Exact CAD motor samples do not establish a relevant velocity ceiling. Failed attempts and the cloned 0.15 scope annotation are preserved with an erratum. Full-CAD videos were already fulfilled; no new recording is claimed. Next verify retention stability and current-state flight/objective response. The same three-hour heartbeat remains active with target 8,192, reserve 300 MiB, host fallback, 180 healthy seconds and all independent gates. The computer stays on.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 text=p.read_text(encoding='utf-8-sig');marker='<!-- adaptive_check_20261005_0841 -->';assert marker in text
 p.write_text(text[:text.index(marker)]+entry,encoding='utf-8')
print('Polished only this check current notes; older chronological history retained.')
