"""Diagnostic training adaptation only; original signature does not certify it."""
import sys,runpy,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(OUT),str(TRAIN),str(TRAIN.parent)]
from shoulder_mean_refinement import ShoulderMeanRefinement
from upper_body_refinement import UpperBodyRefinement
apply='--apply-shoulder-mean' in sys.argv
if apply:sys.argv.remove('--apply-shoulder-mean')
original_init=UpperBodyRefinement.__init__;original_read=UpperBodyRefinement.read_audit;original_close=UpperBodyRefinement.close
def initialize(self,actor,joint_names,*args,**kwargs):
    original_init(self,actor,joint_names,*args,**kwargs)
    self.experimental=ShoulderMeanRefinement(actor,joint_names,apply=apply)
def read(self):
    result=original_read(self);result['experimental_shoulder_mean']=self.experimental.read_audit();return result
def close(self):self.experimental.close();original_close(self)
UpperBodyRefinement.__init__=initialize;UpperBodyRefinement.read_audit=read;UpperBodyRefinement.close=close
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
