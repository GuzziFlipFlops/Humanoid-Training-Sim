"""Exercise a real held Windows read handle, rather than only mocking retry."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from status_io import atomic_json


class StatusIO(unittest.TestCase):
    def test_complete_json_replaces_old_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            atomic_json(path, {'iteration': 1})
            atomic_json(path, {'iteration': 2, 'payload': list(range(200))})
            self.assertEqual(json.loads(path.read_text())['iteration'], 2)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing handles')
    def test_read_handle_cannot_kill_training_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            atomic_json(path, {'iteration': 1})
            reader = path.open('r')
            failures = []
            def update():
                try:
                    atomic_json(path, {'iteration': 2}, timeout=2.)
                except BaseException as error:
                    failures.append(error)
            thread = threading.Thread(target=update)
            thread.start()
            threading.Event().wait(.2)
            reader.close()
            thread.join(timeout=3.)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(json.loads(path.read_text())['iteration'], 2)

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing handles')
    def test_permanent_lock_preserves_old_file_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            atomic_json(path, {'iteration': 1})
            with path.open('r'):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {'iteration': 2}, timeout=.15)
            self.assertEqual(json.loads(path.read_text())['iteration'], 1)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
