"""End-to-end check of lab mode on one machine. Run: python3 test_lab.py"""
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELAY = str(HERE / "gpu_relay.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def watch(p):
    p.lines = queue.Queue()
    p.seen = []
    threading.Thread(target=lambda: [p.lines.put(l) for l in p.stdout], daemon=True).start()
    return p


def start_worker(tmp, name, port, lab=True):
    p = subprocess.Popen([sys.executable, RELAY, "worker", "--name", name, "--host", "127.0.0.1", "--port", str(port),
                          "--jobs", str(tmp / name / "jobs")] + (["--lab"] if lab else []),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "GPU_RELAY_TOKEN": "lab-token", "PYTHONUNBUFFERED": "1"})
    watch(p)
    wait_for(p, "Waiting for jobs")
    return p


def wait_for(proc, text, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            line = proc.lines.get(timeout=0.2)
        except queue.Empty:
            continue
        proc.seen.append(line)
        if text in line:
            return line
    raise AssertionError(f"never printed {text!r}; got:\n{''.join(proc.seen)}")


def drain(*procs):
    for p in procs:
        while not p.lines.empty():
            p.lines.get()


def relay(*args, env):
    return watch(subprocess.Popen([sys.executable, RELAY, *args], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, env=env))


def finish(proc, timeout=60):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        time.sleep(0.3)
        while not proc.lines.empty():
            proc.seen.append(proc.lines.get())
        raise AssertionError("timed out; output:\n" + "".join(proc.seen))
    time.sleep(0.3)
    while not proc.lines.empty():
        proc.seen.append(proc.lines.get())
    return "".join(proc.seen)


def batch_state(folder):
    return {j["n"]: j for j in json.loads((Path(folder) / "batch.json").read_text())["jobs"]}


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    env = {**os.environ, "GPU_RELAY_HOME": str(tmp / "mac"), "GPU_RELAY_NO_BROWSER": "1",
           "GPU_RELAY_LOST_AFTER": "5", "PYTHONUNBUFFERED": "1"}
    project = tmp / "proj"
    project.mkdir()
    (project / "work.py").write_text(
        "import os, sys, time\nn = sys.argv[1]\nprint('nice', os.nice(0) if hasattr(os, 'nice') else 'n/a')\n"
        "for p in (25, 50, 100):\n    print(f'\\rtraining {p}%', end='', flush=True); time.sleep(0.3)\n"
        "os.makedirs('out', exist_ok=True)\nopen(f'out/r{n}.txt', 'w').write('result ' + n)\nprint()\n")
    (project / "slow.py").write_text(
        "import time\nopen('ckpt.txt', 'w').write('checkpoint')\nprint('saved checkpoint', flush=True)\ntime.sleep(120)\n")
    (project / "fail.py").write_text("import sys\nprint('oops')\nsys.exit(3)\n")
    (project / "wait.py").write_text(
        "import time\nprint('working', flush=True)\ntime.sleep(4)\nopen('done.txt', 'w').write('ok')\n")
    jobs = tmp / "jobs.txt"
    jobs.write_text("# one command per line\npython work.py 1\npython work.py 2\npython slow.py\n\npython fail.py\n")
    one_wait = tmp / "wait.txt"
    one_wait.write_text("python wait.py\n")
    one_slow = tmp / "slow.txt"
    one_slow.write_text("python slow.py\n")

    ports = {"lab-a": free_port(), "lab-b": free_port(), "ravi": free_port()}
    workers = {n: start_worker(tmp, n, ports[n], lab=n != "ravi") for n in ports}
    try:
        for n, port in ports.items():
            assert relay("add", n, "127.0.0.1", "lab-token", "--port", str(port), env=env).wait() == 0
        out = finish(relay("friends", "--to", "lab-*", env=env))
        assert out.count("[lab]") == 2 and "ravi" not in out, out

        # 1. A batch of four jobs over two lab PCs: two succeed, one hits the time limit, one fails.
        dash = free_port()
        b = relay("batch", str(project), str(jobs), "--to", "lab-*", "--time-limit", "0.1",
                  "--dashboard-port", str(dash), env=env)
        wait_for(b, "Dashboard:")
        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{dash}/state", timeout=5).read())
        assert {p["name"] for p in state["pcs"]} == {"lab-a", "lab-b"} and len(state["jobs"]) == 4, state
        assert "gpu-relay" in urllib.request.urlopen(f"http://127.0.0.1:{dash}/", timeout=5).read().decode()
        out = finish(b)
        assert b.returncode == 1 and "2 done, 2 not" in out, out
        folder = re.search(r"Results: (.+)", out).group(1).strip()
        jobs_done = batch_state(folder)
        assert [jobs_done[i]["status"] for i in (1, 2, 3, 4)] == ["done", "done", "time limit", "failed"], jobs_done
        assert jobs_done[1]["last"].startswith("training 100%"), jobs_done[1]
        assert jobs_done[4]["code"] == 3
        r1 = list(Path(folder).glob("job001-*/out/r1.txt"))
        assert len(r1) == 1 and r1[0].read_text() == "result 1", r1
        assert list(Path(folder).glob("job003-*/ckpt.txt")), "time-limited job's files did not come back"
        assert not list(Path(folder).glob("job001-*/work.py")), "unchanged files came back"
        log1 = (Path(folder) / "job001.log").read_text()
        if os.name != "nt":
            assert "nice 10" in log1, log1  # jobs run at low priority
        assert "Time limit" in (Path(folder) / "job003.log").read_text()

        drain(*workers.values())
        # 2. A PC shuts down mid-job (worker killed, comes back with no memory): the job runs again.
        b = relay("batch", str(project), str(one_wait), "--to", "lab-a", "--time-limit", "1",
                  "--dashboard-port", str(free_port()), env=env)
        wait_for(workers["lab-a"], "Running job")
        workers["lab-a"].kill()
        workers["lab-a"].wait()
        time.sleep(1)
        workers["lab-a"] = start_worker(tmp, "lab-a", ports["lab-a"])
        out = finish(b)
        assert b.returncode == 0 and "1 done, 0 not" in out, out
        folder = re.search(r"Results: (.+)", out).group(1).strip()
        j = batch_state(folder)[1]
        assert j["tries"] == 1 and j["status"] == "done", j
        assert "will run again" in (Path(folder) / "job001.log").read_text()

        drain(*workers.values())
        # 3. The Mac side is killed mid-batch; --resume picks the still-running job up again.
        b = relay("batch", str(project), str(one_wait), "--to", "lab-b", "--dashboard-port", str(free_port()), env=env)
        folder = re.search(r"Results: (.+)", wait_for(b, "Results:")).group(1).strip()
        wait_for(workers["lab-b"], "Running job")
        time.sleep(2.5)  # let the batch record that the job is running
        b.kill()
        b.wait()
        out = finish(relay("batch", "--resume", folder, "--dashboard-port", str(free_port()), env=env))
        assert "1 done, 0 not" in out, out
        assert batch_state(folder)[1]["tries"] == 0 and list(Path(folder).glob("job001-lab-b/done.txt"))

        drain(*workers.values())
        # 4. Ctrl+C on the Mac cancels the running job on the lab PC.
        b = relay("batch", str(project), str(one_slow), "--to", "lab-a", "--dashboard-port", str(free_port()), env=env)
        wait_for(workers["lab-a"], "Running job")
        time.sleep(2.5)
        b.send_signal(signal.SIGINT)
        out = finish(b)
        assert "cancelled 1 running job" in out and "--resume" in out, out
        wait_for(workers["lab-a"], "ended")

        # 5. Plain `send` to a lab PC needs no Enter at all.
        s = relay("send", str(project), "python work.py 9", "--to", "lab-b", env=env)
        out = finish(s)
        assert s.returncode == 0 and "training 100%" in out and "1 result file(s)" in out, out
    finally:
        for w in workers.values():
            w.kill()

print("OK: lab PCs run jobs without Enter at low priority; batch spreads jobs, enforces the time limit and returns "
      "partial files; a PC that shuts down gets its job re-run; --resume and Ctrl+C work; dashboard serves state")
