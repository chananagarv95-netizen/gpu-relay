"""End-to-end check with two workers on one machine. Run: python3 test_relay.py"""
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELAY = str(HERE / "gpu_relay.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_worker(tmp, name, token, timeout):
    port = free_port()
    p = subprocess.Popen([sys.executable, RELAY, "worker", "--name", name, "--host", "127.0.0.1", "--port", str(port),
                          "--jobs", str(tmp / name / "jobs"), "--approve-timeout", str(timeout)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "GPU_RELAY_TOKEN": token, "PYTHONUNBUFFERED": "1"})
    p.lines = queue.Queue()
    threading.Thread(target=lambda: [p.lines.put(l) for l in p.stdout], daemon=True).start()
    p.port = port
    wait_for(p, "Waiting for jobs")
    return p


def wait_for(worker, text, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        try:
            line = worker.lines.get(timeout=0.2)
        except queue.Empty:
            continue
        if text in line:
            return line
    raise AssertionError(f"worker never printed {text!r}")


def answer(worker, text):
    worker.stdin.write(text + "\n")
    worker.stdin.flush()


def relay(*args, env):
    return subprocess.Popen([sys.executable, RELAY, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    env = {**os.environ, "GPU_RELAY_HOME": str(tmp / "mac")}
    project = tmp / "proj"
    project.mkdir()
    (project / "train.py").write_text(
        "import os\nos.makedirs('out', exist_ok=True)\nprint('epoch 1 loss 0.5')\n"
        "open('out/model.txt', 'w').write('weights')\n")
    (project / "data.txt").write_text("unchanged input")

    away = start_worker(tmp, "away", "tok-a", timeout=3)   # never answers
    ravi = start_worker(tmp, "ravi", "tok-r", timeout=30)
    try:
        for args in (("away", "127.0.0.1", "tok-a", "--port", str(away.port)),
                     ("ravi", "127.0.0.1", "tok-r", "--port", str(ravi.port)),
                     ("ghost", "127.0.0.1", "x", "--port", str(free_port())),
                     ("badtoken", "127.0.0.1", "nope", "--port", str(ravi.port))):
            assert relay("add", *args, env=env).wait() == 0

        out = relay("friends", env=env).communicate()[0]
        rows = {line.split()[0]: line for line in out.splitlines()}
        assert "online" in rows["away"] and "online" in rows["ravi"], out
        assert "offline" in rows["ghost"] and "wrong token" in rows["badtoken"], out

        # 1. Ask everyone: ravi accepts, away's request is withdrawn, results come back.
        send = relay("send", str(project), "python train.py", env=env)
        wait_for(away, ">>> Run this job")
        wait_for(ravi, ">>> Run this job")
        answer(ravi, "")
        wait_for(away, "withdrawn")
        wait_for(ravi, ">>> Job finished")
        answer(ravi, "")
        out = send.communicate(timeout=30)[0]
        assert send.returncode == 0, out
        assert "ravi accepted" in out and "epoch 1 loss 0.5" in out, out
        results = list((project / "relay-results").glob("*/out/model.txt"))
        assert len(results) == 1 and results[0].read_text() == "weights", results
        job_dir = results[0].parent.parent
        assert not (job_dir / "train.py").exists() and not (job_dir / "data.txt").exists(), "unchanged files came back"
        status = relay("friends", env=env).communicate()[0]
        assert "away" in status and "busy" not in status, status

        # 2. A friend says no.
        send = relay("send", str(project), "python train.py", "--to", "ravi", env=env)
        wait_for(ravi, ">>> Run this job")
        answer(ravi, "n")
        out = send.communicate(timeout=30)[0]
        assert send.returncode != 0 and "declined" in out and "Nobody accepted" in out, out

        # 3. A friend who is away: no answer within the timeout counts as no.
        send = relay("send", str(project), "python train.py", "--to", "away", env=env)
        out = send.communicate(timeout=30)[0]
        assert send.returncode != 0 and "no answer" in out, out
    finally:
        away.kill()
        ravi.kill()

print("OK: friends list (online/offline/wrong token); first to accept wins and others are withdrawn; "
      "logs + only changed files returned; decline and no-answer both refused")
