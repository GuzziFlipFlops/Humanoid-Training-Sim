import runpy,sys
from pathlib import Path
TRAIN=Path('E:\\Aditya\\robot_sim\\physical_v29\\training\\runs\\locomotion_20261010_r41_command_conditioned_advantage\\source\\physical_v29\\training')
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
