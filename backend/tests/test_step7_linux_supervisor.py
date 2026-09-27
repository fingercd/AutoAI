"""Linux kernel process proofs, including detached descendants and lost guardian."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from backend.app.runs.supervisor import ChildExecutionError, SupervisionUncertain, supervise

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux subreaper contract')


def deadline():
    return datetime.now(timezone.utc) + timedelta(seconds=10)


def test_detached_orphan_is_reaped_before_failure(tmp_path, monkeypatch):
    module = tmp_path/'orphan_probe.py'
    identity = tmp_path/'identities.json'
    module.write_text('''import sys,json,os,subprocess
from pathlib import Path
p=json.loads(sys.stdin.readline())
c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
Path(p['identity']).write_text(json.dumps(dict(child=os.getpid(),grandchild=c.pid)))
print(json.dumps(dict(ok=True,result=dict(value=1))))
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH',''))
    with pytest.raises(ChildExecutionError) as caught:
        supervise({'identity':str(identity)}, module='orphan_probe',
                  deadline_at=deadline(), active=lambda:True, poll_seconds=.02)
    assert caught.value.error_type == 'ChildTreeStillActive'
    ids=json.loads(identity.read_text())
    assert all(not Path(f'/proc/{pid}').exists() for pid in ids.values())


def test_lost_guardian_never_confirms_exit(tmp_path):
    import ctypes
    libc=ctypes.CDLL(None)
    previous=ctypes.c_int()
    assert libc.prctl(37,ctypes.byref(previous),0,0,0)==0
    assert libc.prctl(36,1,0,0,0)==0
    identity = tmp_path/'identities.json'
    killed = False
    identities = {}
    def active():
        nonlocal killed, identities
        if identity.exists() and not killed:
            identities=json.loads(identity.read_text())
            stat=Path(f"/proc/{identities['child']}/stat").read_text().split(') ',1)[1].split()
            guardian=int(stat[1])
            os.kill(guardian,signal.SIGKILL)
            killed=True
        return True
    evidence=[]
    try:
        with pytest.raises(SupervisionUncertain):
            supervise({'probe':'tree','identity_path':str(identity)}, deadline_at=deadline(),
                      active=active,poll_seconds=.02,on_stop=evidence.append)
        assert killed
        assert evidence[0].exited_at is None
        assert evidence[0].latency_seconds is None
    finally:
        # We adopt this fixture's orphans when its guardian is deliberately
        # killed. waitpid confirms ownership before signalling; no PID scan.
        for pid in identities.values():
            try:
                reaped,_=os.waitpid(pid,os.WNOHANG)
                if not reaped:
                    os.kill(pid,signal.SIGKILL)
                    os.waitpid(pid,0)
            except ChildProcessError:pass
        assert libc.prctl(36,previous.value,0,0,0)==0


def test_large_child_result_does_not_block_guardian_pipe():
    value='x'*200000
    assert supervise({'probe':'return','value':value},deadline_at=deadline(),
                     active=lambda:True,poll_seconds=.02)['value']==value
