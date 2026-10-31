"""Prevent host exhaustion even with spare VRAM; preserve optimizer scheduling."""
import unittest
from types import SimpleNamespace
from memory_guard import recycle_reason,ResourceRecycle,restored_learning_rate


class MemoryMaintenance(unittest.TestCase):
    def test_spare_vram_does_not_hide_host_exhaustion(self):
        self.assertEqual(recycle_reason(dict(free_mb=3000,ram_available_mb=1000,
            ram_total_mb=32580,process_private_mb=18000)),'host_ram_below_2048MiB')

    def test_commit_growth_triggers_before_observed_67gb(self):
        self.assertEqual(recycle_reason(dict(ram_available_mb=6000,
            ram_total_mb=32580,process_private_mb=40000)),'worker_commit_above_host_budget')
        self.assertIsNone(recycle_reason(dict(ram_available_mb=6000,
            ram_total_mb=32580,process_private_mb=17000)))

    def test_missing_measurement_is_not_false_exhaustion(self):
        self.assertIsNone(recycle_reason({}))

    def test_completed_update_accounting(self):
        self.assertEqual(ResourceRecycle(23,'host_ram_below_2048MiB',{}).iterations,24)

    def test_worker_restart_preserves_saved_adaptive_rate(self):
        optimizer=SimpleNamespace(param_groups=[{'lr':1e-5}])
        self.assertEqual(restored_learning_rate(optimizer,.0002),1e-5)
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)
        optimizer.param_groups[0]['lr']=.001
        self.assertEqual(restored_learning_rate(optimizer,.0002),.0002)


if __name__=='__main__':unittest.main(verbosity=2)
