"""Create this check's CPU consolidation from the prior verified parser."""
from pathlib import Path
import ast
OUT=Path(__file__).resolve().parent
old=OUT.parent/'adaptive_check_20261005_1746'
text=(old/'finalize_check.py').read_text()
start=text.index('priority=[')
end=text.index('\ndecision=',start)
priority="""priority=[
 'Prioritize actual per-command advantage-weighted exploration and gradient evidence: checkpoint139 now loses 0.35 speed retention in both tests (0.464/0.459 m/s), while 0.0/0.15/0.65/1.0 pass. Compare qualified teacher labels/coverage, lower effective targets, PPO versus teacher gradients, and tracking-kernel credit on slow and running samples. The CPU running-state audit finds 14.53% raw means outside +/-3, with predicted Gaussian clipping 57.8/63.3% at left/right ankle pitch and 39.0/73.2% at left/right ankle roll. Synthetic tail-score tests are not actual PPO evidence. Passively compare actual minibatch ordinary versus clipped-Gaussian CDF scores, real flight credit, entropy and importance ratios. Only stage a narrow training-only estimator after unbiasedness/numerical tests, unchanged inference/physics/full-state load, and matched current-state 64/128-update control with every nominal/pushed/prior/head/direction gate. CAPG is a primary-source research lead, not deployed or proven causal; the old teacher-clipping-equivalence candidate remains rejected.',
 'Preserve current neutral-head and walking state and compare exact matching prior/current gates with the preserved 133 actor and newest actor. Recent true flight rose to 0.236/0.365% before this pause but remains below 1.5% and is seed-sensitive. The 192-robot current-actor timing probe rejects 20/40/60ms advances: zero flight and lower complete return. Existing offsets produce only 0.479% flight and -4.274 return; never deploy these pulses/labels or repeat fixed shifts without new evidence. Box phase smoothing changes full return by too little to repair the tradeoff and reintroduces edge redistribution credit; it is rejected before learning. Inspect full objective and feasible natural gait, not grounded local-return maximization, arbitrary weights, observation spoofs, head rollback or hybrids.',
 'Verify 6144/8192 moving capacity, one actual worker, 55 frozen hashes, native caps and full-state recovery. The 069 VRAM failure preserved 58 genuine updates in memory_recovery.pt and 070 loaded actor/critic/Adam/exploration exactly; supervisor completed-operation count omits those 58. Report completed and recovered updates separately and all actual retained samples/time, excluding both R37 diagnostic pauses. Current healthy ramp requires about 2091 MiB free GPU for the 6144-to-8192 tier, not merely 300 MiB; keep the reserve and 180 continuous healthy training seconds. Earlier 200Hz no-driving-clip evidence through 2.0 is actor-specific; fallen 2.5 spikes are not a ceiling.'
]"""
text=text[:start]+priority+text[end:]
text=text.replace("action='Reject local diagnostic trajectory transfer; resume exact same R37 full-state learner'", "action='Reject fixed timing advances and this box-smoothed force candidate; preserve exact same R37 learner and investigate actual bound-aware gradient variance'")
text=text.replace("pause['checkpoint_sha256']]", "pause['checkpoint_sha256']]") # Identity, retained parser.
text=text.replace("blocks=[];total_samples=active_seconds=worker_seconds=0.", """blocks=[];total_samples=active_seconds=worker_seconds=0.
pauses=[]
for receipt in HERE.glob('diagnostics/*/diagnostic_resume.json'):
 p=receipt.parent/'diagnostic_pause.json'
 if not p.exists():continue
 try:
  a=read(p);b=read(receipt)
  if Path(a['run']).resolve()==run.resolve() and a['supervisor_pid']==s['supervisor_pid'] and a['paused_unix']>=s['started_unix']:
   pauses.append(dict(receipt=str(receipt),pause_seconds=b['pause_seconds']))
 except (KeyError,ValueError):continue
excluded_pause=sum(p['pause_seconds'] for p in pauses)
""")
text=text.replace("live['recorded_unix']-s['started_unix']-resume['pause_seconds']", "live['recorded_unix']-s['started_unix']-excluded_pause")
text=text.replace("excluded_diagnostic_pause_seconds=resume['pause_seconds']", "excluded_diagnostic_pause_seconds=excluded_pause,documented_pauses=pauses")
needle="write(OUT/'resumed_learning_verification.json',rates)"
replacement="""# Failed-operation updates may be retained in a full memory-recovery checkpoint.
recovered=[]
for a in s['attempts']:
 f=Path(a['folder'])
 if not a.get('memory_failure') or a['operation']!='train' or not (f/'progress.jsonl').exists():continue
 cp=f/'memory_recovery.pt'
 if not cp.exists():continue
 digest=sha(cp)
 receivers=[b for b in blocks if b['full_load'].get('source_sha256')==digest]
 assert receivers, 'Do not count a failed trace as retained without exact subsequent full-state load'
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 recovered.append(dict(folder=str(f),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),
  active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],
  checkpoint=str(cp),checkpoint_sha256=digest,exact_full_state_receivers=[b['folder'] for b in receivers],
  min_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),scope='Guard saved recovery and raised; completed-operation supervisor counter omitted these preserved updates'))
rates['recovered_failed_operation_updates']=sum(a['updates'] for a in recovered)
rates['recovery_evidence']=recovered
rates['actual_retained_main_updates']=rates['completed_main_updates']+rates['recovered_failed_operation_updates']
samples_all=total_samples+sum(a['samples'] for a in recovered)
rates['actual_retained_main_samples']=int(samples_all)
rates['all_retained_active_samples_s']=samples_all/(active_seconds+sum(a['active_seconds'] for a in recovered))
rates['all_retained_worker_effective_samples_s']=samples_all/(worker_seconds+sum(a['worker_seconds'] for a in recovered))
rates['all_retained_whole_chain_samples_s']=samples_all/(live['recorded_unix']-s['started_unix']-excluded_pause)
rates['scope']='Completed-main and recovered-but-failed-operation samples are separate. All retained counts include verified full-state recovery; both R37 diagnostic pauses excluded from whole-chain clocks. Prior R35/R36 pauses predate this run. Paused pushed138 wrapper includes this pause and is not physics throughput.'
write(OUT/'resumed_learning_verification.json',rates)"""
assert needle in text;text=text.replace(needle,replacement)
ast.parse(text)
(OUT/'finalize_check.py').write_text(text)
print('Prepared CPU finalizer; no simulator or training changes')
