"""Record already-inspected slow-bias plots, with exact content hashes."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
p=OUT/'plot_qa.json';qa=json.loads(p.read_text(encoding='utf-8-sig'))
for folder,note in [('candidate_upper_128','Candidate128 inspected: held opposite2.5 shoulder bias and large slow3m reversal; smaller faster residual. No natural-motion certification.'),('final_upper','Original257 inspected: nearly held opposite2.5 shoulder pitch biases; 3m opposite slow reversals dominate angle span. Full-population decomposition saved; representative plot is descriptive only.')]:
    f=OUT/folder/'upper_bias_decomposition.png'
    qa['files'].append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visually_inspected=True,inspection_note=note))
qa['recorded_utc']=datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(dict(inspected_bias_plots=2)))
