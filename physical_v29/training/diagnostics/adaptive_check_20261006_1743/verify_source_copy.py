import json,hashlib
from pathlib import Path
out=Path(__file__).resolve().parent;run=Path(json.loads((out/'diagnostic_pause.json').read_text())['run'])
cfg=json.loads((run/'chain_config.json').read_text());source=run/'source/physical_v29';copy=out/'control_source/physical_v29';records={}
for name,h in cfg['sources_sha256'].items():
 if name in ['supervisor_unitree.py','vram_ramp.py']:continue
 rel=Path('assets')/name.removeprefix('physical_assets/') if name.startswith('physical_assets/') else Path(name.removeprefix('physical_v29/')) if name.startswith('physical_v29/') else Path('training')/name
 a,b=source/rel,copy/rel
 assert hashlib.sha256(a.read_bytes()).hexdigest()==h
 assert b.read_bytes()==a.read_bytes(),name
 records[name]=h
(out/'protected_source_copy.json').write_text(json.dumps({'all_protected_runtime_files':records,'identical_count':len(records),'scope':'All53 frozen physical/runtime inventory files reproduced; two supervisor-only files remain untouched in original run. Generated robot USD is rebuilt from protected assets; protected local ground restored verbatim.'},indent=2)+'\n')
print(len(records))
