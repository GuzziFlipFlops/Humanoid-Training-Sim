from pathlib import Path
out=Path(__file__).resolve().parent;old=out.parent/'adaptive_check_20261006_1243'
t=(old/'matched_exposure_test.py').read_text()
t=t.replace("root=OUT/'matched_exposure'","root=OUT/'matched_direction'")
t=t.replace("source_signature_unchanged=True,truthful_commands=True", "original_authored_signature_unchanged=True,executed_reward_kernel_changed=True,original_signature_does_not_certify_override=True,truthful_commands=True")
t=t.replace("Uniform +/-0.1m/s lateral command only on sampled frontier x>3m/s at stage3.5. Existing stopped/prior exposure, reward, PPO, upper/teacher gradients, native caps and inference are unchanged. Extra sampling changes subsequent random draw stream.", "Fixed-reset-heading linear velocity scoring for frontier x>3 only; original exponential std/weight and all <=3 reward cases unchanged. All physical/action/history/PPO/teachers and independent classifiers unchanged. This explicit runtime reward override is diagnostic and cannot be deployed under the original signature.")
t=t.replace('lateral_exposure_candidate.py','direction_reward_candidate.py').replace("'steering_probe.py'","'arm_response_probe.py'")
t=t.replace("timeout=320","timeout=360")
t=t.replace("if time.time()-start>1600","if time.time()-start>2000")
(out/'matched_direction_test.py').write_text(t)
print('Prepared explicit full-state 64/128 diagnostic comparison; no production edits.')
