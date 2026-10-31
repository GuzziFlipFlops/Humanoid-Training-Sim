"""Prepare tested CPU analysis utilities without touching frozen production."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
old=OUT.parent/'adaptive_check_20261007_1845'
s=(old/'analyze_current_gradients.py').read_text(encoding='utf-8').replace("OUT/'gradient_smoke/run_config.json'","OUT/'matched_focused_prior/passive_PPO_capture/run_config.json'")
(OUT/'analyze_current_gradients.py').write_text(s,encoding='utf-8')
old=OUT.parent/'adaptive_check_20261008_1446'
(OUT/'gate_com_contact_audit.py').write_text((old/'gate_com_contact_audit.py').read_text(encoding='utf-8'),encoding='utf-8')
print('Prepared passive gradient and synchronous COM/contact analyses.')
