"""Counterexamples for the independent landing gate, without importing Isaac."""
import unittest,torch
from evaluation_steps import IndependentLandings


class IndependentGait(unittest.TestCase):
    def setUp(self):
        self.v=IndependentLandings(1,'cpu');self.p=torch.tensor([[[0.,-.12],[0.,.12]]])
        self.r=torch.zeros(1,2);self.h=torch.zeros(1,2);self.command=torch.tensor([1.5])
        self.total=0.;self.hops=0.;self.touches=0.;self.repeated=0.

    def tick(self,c,alive=True):
        events,clearance,hop,touch=self.v.update(torch.tensor([c]),self.p,self.h,self.r,
            torch.tensor([[1.,0.]]),self.command,torch.tensor([alive]),.02)
        self.total+=float(events.sum());self.hops+=float(hop)
        self.touches+=float(touch);self.repeated+=float(self.v.same_foot_events)

    def air(self):
        for _ in range(4):self.tick([True,True])
        self.h[:]=.04
        for _ in range(8):self.tick([False,False])
        self.p[:,:,0]=.15;self.r[:,0]=.10;self.h[:]=0.

    def test_paired_hop_fails(self):
        self.air()
        for _ in range(5):self.tick([True,True])
        self.assertEqual(self.total,0.);self.assertEqual(self.hops,2.)

    def test_staggered_hop_fails(self):
        self.air();self.tick([True,False])
        for _ in range(5):self.tick([True,True])
        self.assertEqual(self.total,0.);self.assertEqual(self.hops,2.)

    def test_alternating_run_landing_passes(self):
        self.air()
        for _ in range(8):self.tick([True,False])
        for _ in range(5):self.tick([True,True])
        self.assertEqual(self.total,2.);self.assertEqual(self.hops,0.)

    def test_airborne_in_place_kick_fails(self):
        self.air();self.p[:,:,0]=0.
        for _ in range(8):self.tick([True,False])
        self.assertEqual(self.total,0.)

    def test_slide_after_rejected_landing_does_not_pay(self):
        self.air();self.p[:,:,0]=0.
        for _ in range(5):self.tick([True,False])
        self.p[:,0,0]=.20
        for _ in range(8):self.tick([True,False])
        self.assertEqual(self.total,0.)

    def test_overlapping_hop_pairs_count_each_touch_once(self):
        self.air()
        for _ in range(3):self.tick([True,False])
        for _ in range(4):self.tick([False,True])
        self.tick([True,True])
        self.assertEqual(self.touches,3.)
        self.assertEqual(self.hops,3.)
        self.assertLessEqual(self.hops/self.touches,1.)

    def step(self,side):
        c=[True,True];c[side]=False
        self.h[:,side]=.04;self.tick(c)
        self.p[:,side,0]+=.08;self.r[:,0]+=.02
        for _ in range(4):self.tick(c)
        self.h[:]=0.
        for _ in range(5):self.tick([True,True])

    def test_repeated_same_foot_is_measured_and_rejected_by_order_gate(self):
        from test_gates import GATE,good_stats
        self.command[:]=.15
        for _ in range(5):self.tick([True,True])
        for _ in range(8):self.step(0)
        self.assertEqual(self.total,8.);self.assertEqual(self.repeated,7.)
        stats=good_stats(.15);stats['same_foot_fraction']=self.repeated/self.total
        self.assertIn('alternating_landings',GATE(.15,stats)['failures'])

    def test_fast_alternating_physical_landings_are_not_filtered_to_fake_cadence(self):
        from test_gates import GATE,good_stats
        self.command[:]=.15
        for _ in range(5):self.tick([True,True])
        start=self.v.time
        for side in [0,1]*10:self.step(side)
        self.assertEqual(self.total,20.);self.assertEqual(self.repeated,0.)
        stats=good_stats(.15);stats['cycle_hz']=self.total/2/(self.v.time-start)
        self.assertIn('gait_cadence',GATE(.15,stats)['failures'])

    def test_slide_during_landing_dwell_does_not_change_impact_travel(self):
        self.air();self.p[:,:,0]=0.;self.tick([True,False])
        self.p[:,0,0]=.20
        for _ in range(8):self.tick([True,False])
        self.assertEqual(self.total,0.)

    def test_after_first_fall_cannot_pay_new_events(self):
        self.air()
        for _ in range(8):self.tick([True,False],alive=False)
        self.assertEqual(self.total,0.)


if __name__=='__main__':unittest.main(verbosity=2)
