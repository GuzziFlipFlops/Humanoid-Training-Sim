import importlib.util,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent
old=out.parent/'adaptive_check_20261006_1243'
for name in ['review_gates.py','upper_joint_time_audit.py','summarize_training.py']:
    shutil.copy2(old/name,out/name)
p=out.parents[1]/'record_owned_observation_handoff.py'
s=importlib.util.spec_from_file_location('owned_recorder',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.OUT=out;m.main()
shutil.copy2(Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml'),out/'automation_before.toml')
