"""Record user-facing monitor evidence and physical-plot inspection."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
assert read(OUT/'neutral_teacher_response_analysis.json')['completed']
(OUT/'physical_visual_inspection.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
 plot=str(OUT/'neutral_teacher_response.png'),sha256=hashlib.sha256((OUT/'neutral_teacher_response.png').read_bytes()).hexdigest(),visually_inspected=True,
 observations='Both physical seeds nearly agree. Current1112 has near-zero1.5 genuine flight,267 body/arms with current head about2.36-2.39% and higher complete reward. Whole-upper composite now has about1.6% flight but lower complete return and larger slow speed; do not repeat the dated claim that every whole-upper test is below1.5%. These are diagnostic closed-loop composites,not deployment or pure mechanical isolation. Slow head motion remains a separate cost.',
 reused_previous_teacher_certificate=True,new_current_actor_physical_test=True),indent=2)+'\n')
