"""Save the repaired source's targeted unit-test results for the handoff."""
import json
import sys
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAINING = OUT.parents[1]
sys.path.insert(0,str(TRAINING));sys.path.insert(0,str(TRAINING.parent))
names=['test_heading_alignment','test_heading_provenance','test_heading_direction',
       'test_standing_retention','test_memory_guard','test_vram_ramp','test_cad_differential_cache']
with (OUT/'cpu_tests.log').open('w') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),modules=names,
    scope='Targeted reward ranking, protected code/gates, transfer, retention gradient, capacity and CAD regression tests; not gait qualification.')
(OUT/'cpu_tests.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));assert result.wasSuccessful()
