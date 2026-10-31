"""Diagnostic ONLY: shoulder pitch K70, all native caps and other gains unchanged.
Original signature does not certify this physical gain override. No diagnostic
learned actor is deployable without versioned physics and explicit migration.
"""
import builtins,runpy,sys
from pathlib import Path
out=Path(__file__).resolve().parent;train=out/'control_source/physical_v29/training'
sys.path[:0]=[str(train),str(train.parent)]
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
 global installed
 result=original_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True
  cls=sys.modules[name].LocomotionEnv;original_init=cls.__init__
  def initialize(self,*a,**kw):
   original_init(self,*a,**kw)
   ids=[self.lookup[x] for x in ['left_shoulder_pitch','right_shoulder_pitch']]
   assert (self.actuator.stiffness[:,ids]==35.).all() and (self.actuator.damping[:,ids]==1.).all()
   self.actuator.stiffness[:,ids]=70.
   assert (self.actuator.stiffness[:,ids]==70.).all()
  cls.__init__=initialize
  builtins.__import__=original_import
 return result
builtins.__import__=install
runpy.run_path(str(train/'worker_unitree.py'),run_name='__main__')
