"""Sequential single-worker diagnostic, with bounded work and owned resume."""
import json, subprocess, sys, time, traceback, hashlib, shutil
from pathlib import Path
import psutil
OUT = Path(__file__).resolve().parent
ROOT = OUT / 'bounded_diagnosis'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
NAMES = {'worker_unitree.py', 'arm_target_probe.py', 'capture_current_gradients.py'}
WORK_START = None

def kill_owned(proc, created):
    try:
        if proc.is_running() and proc.create_time() == created:
            children = proc.children(recursive=True)
            tree = children + [proc]
            for p in tree:
                if p.is_running(): p.terminate()
            _, alive = psutil.wait_procs(tree, timeout=15)
            for p in alive: p.kill()
            psutil.wait_procs(alive, timeout=15)
    except psutil.NoSuchProcess: pass

def launch(argv, folder, timeout):
    assert time.monotonic() - WORK_START < 1400, 'Work-only diagnostic budget exhausted'
    folder.mkdir(exist_ok=False)
    launched_source = Path(argv[0])
    shutil.copyfile(launched_source, folder / 'executed_source.py')
    (folder / 'source_receipt.json').write_text(json.dumps(dict(path=str(launched_source),
        sha256=hashlib.sha256(launched_source.read_bytes()).hexdigest(),
        original_frozen_sources_unchanged=True, external_diagnostic_controller_only=True), indent=2))
    for p in psutil.process_iter(['cmdline']):
        assert not any(Path(x).name in NAMES for x in p.info['cmdline'] or []), 'Concurrent CUDA diagnostic'
    start = time.time()
    with (folder / 'stdout.log').open('w', encoding='utf-8') as stdout, (folder / 'stderr.log').open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen([sys.executable, '-u', '-X', 'utf8'] + argv, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW)
        identity = psutil.Process(child.pid); created = identity.create_time()
        (folder / 'process.json').write_text(json.dumps(dict(pid=child.pid, create_time=created, argv=argv), indent=2))
        deadline = time.monotonic() + timeout
        try:
            while child.poll() is None:
                if launched_source.name == 'arm_target_probe.py' and (OUT / 'arm_probe_failure.json').exists():
                    raise RuntimeError('Physical diagnostic recorded a failure; close only this owned worker and resume safely.')
                assert time.monotonic() < deadline, 'Diagnostic operation budget exhausted'
                assert psutil.virtual_memory().available / 1024**2 > 2000, 'Diagnostic host memory reserve breached'
                time.sleep(2)
            assert child.returncode == 0 and read(folder / 'result.json')['completed'], (child.returncode, str(folder))
        except BaseException:
            kill_owned(identity, created); raise
    receipt = dict(folder=str(folder), wall_seconds=time.time() - start, exit_code=child.returncode,
        work_seconds=time.monotonic() - WORK_START, completed=True)
    (folder / 'operation_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)

ROOT.mkdir(exist_ok=False)
try:
    with (OUT / 'pause.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-X', 'utf8', str(OUT / 'prepare_and_pause.py')], check=True, stdout=log, stderr=subprocess.STDOUT)
    WORK_START = time.monotonic()
    pause = read(OUT / 'diagnostic_pause.json'); cp = pause['checkpoint']
    launch([str(OUT / 'arm_target_probe.py'), '--headless', '--device', 'cuda:0'], OUT / 'arm_target_response', 780)
    result=read(OUT/'arm_target_response/result.json')
    assert any(g['mode']=='body_replay_zero' for g in result['groups'])
    assert result['maximum_physical_target_reconstruction_error_rad']<1e-6
    (ROOT/'validation.json').write_text(json.dumps(dict(passed=True,explicit_zero_offset_replay=True,
        diagnostic_controllers_not_deployed=True,work_seconds=time.monotonic()-WORK_START),indent=2))

except BaseException:
    (ROOT / 'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),
        work_seconds=time.monotonic() - WORK_START if WORK_START else None), indent=2))
    raise
finally:
    if (OUT / 'diagnostic_pause.json').exists():
        with (OUT / 'resume.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(OUT / 'resume_after_diagnosis.py')],
                check=True, stdout=log, stderr=subprocess.STDOUT)
print('Bounded diagnostics finished; exact owned learner resumed.', flush=True)
