"""Diagnostic only: truthful labels for the certified neutral-head 349 teacher.

Inference observations, physical actions, PPO, all weights and upper auxiliary
remain original. The inherited signature DOES NOT certify this training override.
"""
import sys,runpy,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
from upper_body_refinement import LocomotionOnlyRetention
original=LocomotionOnlyRetention.__init__
def initialize(self,*a,**kw):
 original(self,*a,**kw)
 if self.command_range is not None:
  assert self.command_range==(.1,3.0),'Candidate is tied to the current qualified neutral-head teacher'
  assert '--locomotion_prior_certificate' in sys.argv
  certificate=json.loads(Path(sys.argv[sys.argv.index('--locomotion_prior_certificate')+1]).read_text())
  assert Path(certificate['checkpoint']).parent.name=='349_run_300_train_018'
  assert hashlib.sha256(Path(certificate['checkpoint']).read_bytes()).hexdigest()=='18e19ec3b558922a5927136d2b1b51fc3b68dc44a22df252e5630fce84c00a3e'
  self.phase_tables={}
LocomotionOnlyRetention.__init__=initialize
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
