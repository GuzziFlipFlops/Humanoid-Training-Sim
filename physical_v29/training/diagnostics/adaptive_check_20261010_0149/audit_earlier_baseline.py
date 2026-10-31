"""Fresh earlier267 all-command upper audit; saved gates, CPU only."""
from pathlib import Path
import json,hashlib,subprocess,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text())
folders={k:ROOT/('earlier_baseline_'+s) for k,s in [('nominal','nominal'),('pushed','push')]}
assert all((p/'operation_receipt.json').exists() for p in folders.values())
cp=Path(read(folders['nominal']/'run_config.json')['arguments']['checkpoint'])
pair=dict(checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),**{k:str(p) for k,p in folders.items()})
gate=OUT/'earlier_baseline_gate_review.json'
gate.write_text(json.dumps(dict(latest=pair,promotions=[],run=read(OUT/'diagnostic_pause.json')['run']),indent=2)+'\n')
subprocess.run([sys.executable,'-X','utf8',str(OUT/'run_upper_audit_set.py'),'--gate-review',str(gate),'--output','earlier_baseline_upper','--all-traces'],check=True)
