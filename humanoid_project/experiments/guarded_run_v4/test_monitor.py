import unittest
from unittest.mock import patch
from pathlib import Path
from monitor import duration,eta_to,log_progress,render

class Tests(unittest.TestCase):
    def test_count_and_rate(self):
        progress,rate=log_progress('Learning iteration 69/100\nIteration time: 4.0s\nIteration time: 6.0s')
        self.assertEqual(progress,(70,100)); self.assertEqual(rate,5.)
    def test_eta_includes_intermediate_not_final_eval(self):
        self.assertEqual(eta_to(100,70,100,4.,200.,30.),120.)
        self.assertEqual(eta_to(200,70,100,4.,200.,30.),750.)
        self.assertEqual(eta_to(800,70,100,4.,200.,30.),4530.)
    def test_pause_and_200_blocks(self):
        self.assertEqual(eta_to(200,100,100,4.,200.,30.,80.),480.)
        self.assertEqual(eta_to(400,100,200,4.,200.,30.),1430.)
        self.assertEqual(eta_to(800,800,100,None,200.,30.),0.)
        self.assertIsNone(eta_to(100,0,100,None,200.,30.))
    def test_missing_and_long_times(self):
        self.assertEqual(log_progress('starting'),((0,0),None))
        self.assertEqual(duration(3601),'1h 00m 01s')
    def test_stopped_eval_uses_stage_not_block_count(self):
        s=dict(status='stopped_gate',stage='balance',stage_budget=800,num_envs=8192,
               active_block='00_balance_1_0600_eval_83')
        with patch('monitor.tail',return_value='Learning iteration 99/100\nIteration time: 4.0s'), patch('monitor.read_json',return_value=None):
            output=render(s,Path('nonexistent_monitor_fixture'))
        self.assertIn('600 / 800',output)
        self.assertIn('No active-training ETA',output)
    def test_training_outputs_both_milestones(self):
        s=dict(status='training',stage='balance',stage_budget=800,num_envs=8192,
               active_block='00_balance_1_0100_8192',stage_completed_before_block=0)
        with patch('monitor.tail',return_value='Learning iteration 69/100\nIteration time: 4.0s'), patch('monitor.read_json',return_value=None):
            output=render(s,Path('nonexistent_monitor_fixture'))
        self.assertIn('70 / 800',output)
        self.assertIn('checkpoint at iteration 100',output)
        self.assertIn('milestone (200)',output)

if __name__=='__main__': unittest.main(verbosity=2)
