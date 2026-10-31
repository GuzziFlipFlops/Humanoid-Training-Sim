"""Refresh owned process/source evidence without allocating a simulator."""
import importlib.util, argparse
from pathlib import Path
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owned_recorder',OUT.parents[1]/'record_owned_observation_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--folder',default='closing_cpu');args=p.parse_args()
m.OUT=OUT/args.folder
m.main()
