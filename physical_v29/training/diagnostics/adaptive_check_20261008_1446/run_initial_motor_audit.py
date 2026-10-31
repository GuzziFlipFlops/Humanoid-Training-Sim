"""Reuse tested exact CAD mapping with current, interleaved case provenance."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
source=OUT.parent/'adaptive_check_20261008_0946/audit_current_motors.py'
script=source.read_text().replace("OUT/'closing_gate_review.json'","OUT/'initial_gate_review.json'")
(OUT/'audit_current_motors.py').write_text(script)
exec(compile(script,str(OUT/'audit_current_motors.py'),'exec'),{'__file__':str(OUT/'audit_current_motors.py'),'__name__':'__main__'})
