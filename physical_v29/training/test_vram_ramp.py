import unittest
from vram_ramp import VramRamp, estimate_increment, training_iterations


class RampTests(unittest.TestCase):
    def feed(self,r,count=3072,free=1500,start=0,seconds=180,ready=True):
        for t in range(start,start+seconds+1,3): r.observe(t,count,free,ready)

    def test_full_three_minutes_required(self):
        r=VramRamp();self.feed(r,seconds=177);self.assertIsNone(r.ready)
        r.observe(180,3072,1500);self.assertEqual(r.consume(),4096)
        self.assertIsNone(r.consume())

    def test_low_free_resets_even_a_ready_decision(self):
        r=VramRamp();self.feed(r);r.observe(183,3072,700);self.assertIsNone(r.ready)
        self.feed(r,start=186,seconds=177);self.assertIsNone(r.ready)

    def test_eval_or_startup_never_qualifies(self):
        r=VramRamp();self.feed(r,ready=False,seconds=360);self.assertIsNone(r.ready)

    def test_missing_sample_or_long_gap_resets(self):
        r=VramRamp();self.feed(r,seconds=177);r.observe(195,3072,1500);self.assertIsNone(r.ready)
        r.observe(198,3072,None);self.assertIsNone(r.since)
        r.observe(201,3072,float('nan'));self.assertIsNone(r.since)

    def test_memory_for_increment_plus_exact_reserve(self):
        r=VramRamp();s=r.observe(0,4096,1323);self.assertEqual(s['required_free_mb'],1324)
        self.assertIsNone(r.since)
        r.observe(3,4096,1324);self.assertEqual(r.since,3)

    def test_count_change_and_ceiling_reset(self):
        r=VramRamp();self.feed(r);r.observe(183,4096,3000);self.assertIsNone(r.ready)
        self.feed(r,count=8192,start=186,seconds=360);self.assertIsNone(r.ready)

    def test_failure_cooldown_and_cost_increase(self):
        r=VramRamp();self.feed(r);r.memory_failure(180)
        self.assertIsNone(r.ready);self.assertEqual(r.mib_per_env,.625)
        self.feed(r,start=183,seconds=174);self.assertIsNone(r.since)
        self.feed(r,start=360);self.assertEqual(r.ready,4096)

    def test_empirical_increment_is_conservative(self):
        b=[dict(environments=4096,minimum_free_mb=1518),dict(environments=6144,minimum_free_mb=786)]
        self.assertEqual(estimate_increment(b),.5)
        b[1]['minimum_free_mb']=0
        self.assertGreater(estimate_increment(b),.9)

    def test_host_failure_does_not_inflate_gpu_cost(self):
        r=VramRamp();self.feed(r);r.memory_failure(180,'host')
        self.assertIsNone(r.ready);self.assertEqual(r.mib_per_env,.5)
        self.assertEqual(r.cooldown_until,360)

    def test_host_headroom_is_required_for_whole_hold(self):
        r=VramRamp()
        for t in range(0,181,3):r.observe(t,4096,3000,True,3000)
        self.assertIsNone(r.ready);self.assertFalse(r.last_status['host_healthy'])
        for t in range(183,364,3):r.observe(t,4096,3000,True,6000)
        self.assertEqual(r.ready,6144)
        r.observe(366,4096,3000,True,2000);self.assertIsNone(r.ready)

    def test_fast_blocks_allow_full_allocation_observation(self):
        self.assertEqual(training_iterations(25,6.5,4096),33)
        self.assertGreaterEqual((training_iterations(25,.8,512)-2)*.8,180)
        self.assertEqual(training_iterations(25,12,6144),25)
        self.assertEqual(training_iterations(25,1,8192),25)
        self.assertEqual(training_iterations(25,None,3072),60)


if __name__=='__main__': unittest.main()
