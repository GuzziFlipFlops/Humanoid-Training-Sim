"""Stalled posture, finite discovery and reset/command farming counterexamples."""
import unittest
import torch
from walking_rewards import StepDiscovery, StepTracker, walking_terms


class Discovery(unittest.TestCase):
    def setUp(self):
        self.bank=StepDiscovery(1,'cpu');self.tracker=StepTracker(1,'cpu')
        self.command=torch.tensor([.15]);self.root=torch.zeros(1,2)
        self.feet=torch.tensor([[[0.,-.15],[0.,.15]]]);self.heading=torch.tensor([[1.,0.]])
        self.total=0.

    def tick(self,force=(169.,169.),clearance=(0.,0.),eligible=True,dt=.02,track=False):
        f=torch.tensor([force]);h=torch.tensor([clearance]);live=torch.tensor([eligible])
        events=self.tracker.update(f>16.9,self.feet,h,self.root,self.heading,self.command,live,dt) if track else torch.zeros(1,2)
        rate=self.bank.update(f,self.feet,h,self.root,self.heading,self.command,live,torch.ones(1),events,338.,dt)
        self.total+=float(rate)*dt
        return float(rate)*dt,events

    def settle(self,**kwargs):
        import math
        for _ in range(math.ceil(.10/kwargs.get('dt',.02))):self.tick(**kwargs)

    def test_stationary_planted_pose_earns_no_discovery(self):
        for _ in range(1000):self.tick()
        self.assertEqual(self.total,0.)

    def test_load_rocking_pays_only_once_per_foot(self):
        self.settle()
        for _ in range(500):
            self.tick((338.,0.));self.tick((0.,338.))
        self.assertAlmostEqual(self.total,.4,places=6)

    def test_discovery_remains_available_after_old_time_window(self):
        for _ in range(1000):self.tick()
        self.root[:,0]=.01;self.feet[:,0,0]=.06
        credit,_=self.tick((0.,338.),(.04,0.))
        self.assertAlmostEqual(credit,1.,places=6)
        for _ in range(100):self.tick((0.,338.),(.04,0.))
        self.assertAlmostEqual(self.total,1.,places=6)

    def test_in_place_kicks_have_no_placement_credit_and_cannot_repeat(self):
        self.settle()
        for side in [0,1]*50:
            self.feet[:,side,0]=.10
            self.tick((0.,338.) if side==0 else (338.,0.),(.04,0.) if side==0 else (0.,.04))
            self.settle()
        self.assertAlmostEqual(self.total,1.2,places=6)

    def test_planted_sliding_pays_no_discovery(self):
        self.settle()
        for _ in range(100):
            self.feet[:,:,0]+=.01;self.root[:,0]+=.01;self.tick()
        self.assertEqual(self.total,0.)

    def test_command_resampling_and_stop_restart_do_not_replenish(self):
        self.settle();self.root[:,0]=.01;self.feet[:,0,0]=.06
        self.tick((0.,338.),(.04,0.));self.command[:]=0.;self.settle()
        self.command[:]=.35
        for _ in range(50):self.tick((0.,338.),(.04,0.))
        self.assertAlmostEqual(self.total,1.,places=6)

    def test_flight_and_ineligible_contacts_cannot_pay(self):
        self.settle();self.root[:,0]=.01;self.feet[:,:,0]=.06
        self.tick((0.,0.),(.05,.05))
        self.tick((0.,338.),(.04,0.),eligible=False)
        self.assertEqual(self.total,0.)

    def test_physical_accepted_landing_cannot_renew_discovery(self):
        self.settle(track=True);self.tick((0.,338.),(.04,0.),track=True)
        self.root[:,0]=.01;self.feet[:,0,0]=.06
        for _ in range(5):self.tick((0.,338.),(.04,0.),track=True)
        events=sum((self.tick(track=True)[1] for _ in range(5)))
        self.assertEqual(float(events[0,0]),1.)
        self.assertEqual(float(self.bank.best[0,0]),1.)
        self.assertEqual(float(self.bank.best[0,1]),0.)
        # A same-foot repeat is rejected and cannot renew its new bank again.
        for _ in range(20):self.tick(track=True)
        self.tick((0.,338.),(.04,0.),track=True)
        self.root[:,0]=.02;self.feet[:,0,0]=.12
        for _ in range(5):self.tick((0.,338.),(.04,0.),track=True)
        for _ in range(20):self.tick(track=True)
        self.assertEqual(float(self.bank.best[0,0]),1.)

    def test_many_valid_alternating_steps_pay_at_most_two_discovery_units(self):
        self.settle(track=True);steps=0.
        for side in [0,1]*8:
            force=(0.,338.) if side==0 else (338.,0.)
            height=(.04,0.) if side==0 else (0.,.04)
            self.tick(force,height,track=True)
            self.root[:,0]+=.01;self.feet[:,side,0]+=.06
            for _ in range(5):self.tick(force,height,track=True)
            for _ in range(5):steps+=float(self.tick(track=True)[1].sum())
            for _ in range(15):self.tick(track=True)
        self.assertEqual(steps,16.)
        self.assertAlmostEqual(self.total,2.,places=6)

    def test_only_physical_reset_replenishes_discovery(self):
        self.settle();self.root[:,0]=.01;self.feet[:,0,0]=.06
        self.tick((0.,338.),(.04,0.));self.bank.reset(torch.tensor([0]))
        self.settle();self.root[:,0]+=.01;self.feet[:,0,0]+=.06
        self.tick((0.,338.),(.04,0.))
        self.assertAlmostEqual(self.total,2.,places=6)

    def test_discovery_event_is_control_step_invariant(self):
        for dt in [.01,.02,.04]:
            self.setUp();self.settle(dt=dt)
            self.root[:,0]=.01;self.feet[:,0,0]=.06
            self.tick((0.,338.),(.04,0.),dt=dt)
            self.assertAlmostEqual(self.total,1.,places=6)

    def test_all_free_discovery_is_below_fall_event_cost(self):
        # Old temporary gait/reference/flight + two new per-foot banks.
        # Load-transfer rate now requires a recent physical accepted step.
        for speed in [.081,.12,.15,.35,1.,1.5,2.5]:
            maximum=(2*.8+.35+.25)*(2.5/(.9+.5*speed))+2.
            self.assertLess(maximum,8.)


class MotionDeficit(unittest.TestCase):
    def terms(self,command=.15,healthy=1.,vx=0.,vy=0.,slip=0.,contact=(True,True),recent=False):
        feet=torch.tensor([[[0.,-.15,-.8],[0.,.15,-.8]]])
        return walking_terms(torch.tensor([command]),torch.tensor([[vx,vy,0.]]),torch.tensor([healthy]),
            torch.tensor([slip]),torch.tensor([contact]),torch.tensor([[True,False]]),
            torch.zeros(1,2),feet,feet,torch.zeros(1),torch.zeros(1,2),.02,
            advance_fraction=torch.tensor([1. if vx>0 else 0.]),swing_budget=torch.zeros(1),recent_step=torch.tensor([recent]))

    def test_indefinite_moving_posture_income_is_negative(self):
        result=self.terms()
        rate=.65+.40+.70+float(result['motion_deficit'])+float(result['missing_swing'])
        self.assertLess(rate,0.)
        self.assertAlmostEqual(float(result['posture_scale']),1.)

    def test_ineligibility_cannot_switch_off_the_cost(self):
        self.assertAlmostEqual(float(self.terms(healthy=0.,vx=.15,contact=(True,False))['motion_deficit']),-1.8,places=6)

    def test_stopping_is_not_penalized(self):
        self.assertEqual(float(self.terms(command=0.)['motion_deficit']),0.)

    def test_real_progress_allows_double_support_after_valid_step(self):
        self.assertAlmostEqual(float(self.terms(vx=.15,recent=True)['motion_deficit']),0.,places=6)
        self.assertAlmostEqual(float(self.terms(vx=.15)['motion_deficit']),-1.8,places=6)

    def test_slip_lateral_motion_and_missing_net_travel_still_cost(self):
        self.assertLess(float(self.terms(vx=.15,slip=.2,contact=(True,False))['motion_deficit']),-1.7)
        self.assertLess(float(self.terms(vx=.15,vy=.2,contact=(True,False))['motion_deficit']),-1.7)
        self.assertLess(float(self.terms(vx=0.,contact=(True,False))['motion_deficit']),-1.7)


if __name__=='__main__':unittest.main(verbosity=2)
