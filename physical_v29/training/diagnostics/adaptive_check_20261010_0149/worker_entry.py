"""Execute the identical protected worker from the diagnostic source copy."""
import runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent/'control_source/physical_v29/training'
sys.path[:0]=[str(HERE),str(HERE.parent)]
runpy.run_path(str(HERE/'worker_unitree.py'),run_name='__main__')
