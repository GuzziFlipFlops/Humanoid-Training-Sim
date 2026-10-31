"""Stop only this owned diagnostic GUI; preserve its rejected preview evidence."""
import json,shutil
from pathlib import Path
import psutil
out=Path(__file__).resolve().parent
dest=Path(json.loads((out/'capture_config.json').read_text())['destination'])
started=json.loads((dest/'capture_started.json').read_text())
p=psutil.Process(started['pid']);assert p.create_time()==started['created_time']
assert any(Path(a).name=='record_full_body_gui.py' for a in p.cmdline())
assert not (dest/'manifest.json').exists()
attempt=out/'attempt01_white_preview';attempt.mkdir(exist_ok=True)
for name in ['neutral_preview.png','preview_ready.json','rendered_mesh_inventory.json','capture_started.json']:
    if (dest/name).exists():shutil.copy2(dest/name,attempt/name)
shutil.copy2(out/'record_full_body_gui.py',attempt/'record_full_body_gui.py')
(attempt/'rejection.json').write_text(json.dumps({'reason':'All140 meshes rendered and attached, but material contrast is washed out and dome primary rays remain hidden. Fix display only before capture.','owned_gui_pid':p.pid},indent=2))
p.terminate();p.wait(timeout=30)
print('Owned rejected-preview GUI stopped; training supervisor remains safely suspended')
