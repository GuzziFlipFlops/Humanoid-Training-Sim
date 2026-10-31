"""Reproduce/fix the installed RSL-RL early-save API on CPU, with a real actor."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

HERE=Path(__file__).parent


class EarlySave(unittest.TestCase):
    def test_installed_api_and_actual_worker_initialization(self):
        out=HERE/'diagnostics/r25_run_review_20261003_1919/logger_cpu'
        out.mkdir(parents=True,exist_ok=True)
        saved=torch.load(HERE/'runs/locomotion_20261003_r25_owned_observations/155_run_150_train_003_part_06/candidate.pt',
                         map_location='cpu',weights_only=False)
        runner=object.__new__(OnPolicyRunner)
        runner.alg=NS(save=lambda:copy.deepcopy(saved))
        runner.current_learning_iteration=0
        runner.logger=object.__new__(Logger)
        with self.assertRaisesRegex(AttributeError,'writer'):
            runner.save(str(out/'before.pt'),infos=saved['infos'])
        assignment=next(n for n in ast.walk(ast.parse((HERE/'worker_unitree.py').read_text()))
                        if isinstance(n,ast.Assign) and len(n.targets)==1
                        and ast.unparse(n.targets[0])=='runner.logger.writer')
        self.assertIsNone(assignment.value.value)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),
                     str(HERE/'worker_unitree.py'),'exec'),dict(runner=runner))
        runner.save(str(out/'after.pt'),infos=saved['infos'])
        loaded=torch.load(out/'after.pt',map_location='cpu',weights_only=False)
        self.assertEqual(loaded['infos'],saved['infos'])
        self.assertTrue(all(torch.equal(v,loaded['actor_state_dict'][k])
                            for k,v in saved['actor_state_dict'].items()))


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
