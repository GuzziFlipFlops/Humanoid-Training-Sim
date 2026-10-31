"""Apply the measured-duration ramp correction after draining r9's worker."""
import traceback
import upgrade_to_r9 as bridge
bridge.PRIOR=bridge.HERE/'runs/locomotion_20261001_r9_auto_vram'
bridge.RUN=bridge.HERE/'runs/locomotion_20261001_r10_auto_vram_duration'
if __name__=='__main__':
 try:bridge.main()
 except BaseException:
  bridge.report(status='failed',error=traceback.format_exc());raise
