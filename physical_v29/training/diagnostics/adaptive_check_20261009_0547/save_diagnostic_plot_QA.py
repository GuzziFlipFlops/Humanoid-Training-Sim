"""Record hashes only for diagnostic plots explicitly visually inspected."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent
rows=[]
for folder,notes in [
    ('','Population and paired-seed compatibility plot inspected. Whole current upper on teacher body overspeeds at 0.15 and lowers flight/full reward at 1.5. All cohorts survive. Diagnostic composites are not production policies or all-speed qualification.'),
    ('head_arm_factorial','Population and paired-seed head/arms split plot inspected. Current arms on teacher body/head explain most 0.15 overspeed and larger 1.5 flight loss; current head also changes flight. Both seeds reproduce the direction of effects. This is closed-loop component attribution, not mechanical isolation.')]:
    f=out/folder/'teacher_compatibility.png'
    assert f.is_file()
    rows.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_inspected=True,observations=notes))
(out/'diagnostic_plot_visual_QA.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),physical_actor='917_run_350_train_303',plots=rows),indent=2)+'\n')
print('Recorded two explicitly inspected diagnostic plots.')
