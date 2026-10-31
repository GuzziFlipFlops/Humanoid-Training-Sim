"""Adapt CPU-only evidence tools for this complete-state comparison."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261009_2048'
receipts=[]
for name in ['audit_matched_upper.py','compare_matched_upper.py','compare_saved_gate_trends.py']:
    text=(OLD/name).read_text()
    text=text.replace('matched_row_route','matched_full_state').replace('[251,263]','[317,331]')
    text=text.replace("['control','shared','rows']","['control','earlier']")
    text=text.replace('len(records)==120','len(records)==80').replace('all_six128_actors_complete','all_four128_actors_complete')
    (OUT/name).write_text(text,encoding='utf-8')
    receipts.append(dict(name=name,source_sha256=hashlib.sha256((OLD/name).read_bytes()).hexdigest(),sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
(OUT/'cpu_review_lineage.json').write_text(json.dumps(dict(CPU_only=True,receipts=receipts),indent=2)+'\n')
print('Three CPU review tools prepared; no training or worker source edited.')
