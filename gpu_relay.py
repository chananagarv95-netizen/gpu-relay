"""GPU relay: run projects on friends' GPU laptops (they press Enter to approve)
or on lab PCs you have permission to use (jobs run automatically).

Works on Windows, macOS and Linux (Python 3.9+, standard library only).

Friend's laptop:  gpu-relay worker --name ravi
Lab PC:           gpu-relay worker --lab            (no Enter needed; see lab/install-lab-pc.ps1)
Your computer:    gpu-relay add ravi <address> <token>          (once per friend)
                  gpu-relay add-many <token> lab1-pc01 lab1-pc02 (lab PCs, by Tailscale name)
                  gpu-relay friends                              (who is online)
                  gpu-relay send ./my-project "python train.py"  (one job, first to accept)
                  gpu-relay batch ./my-project jobs.txt --to "lab1-*"  (many jobs, one per free PC)
                  gpu-relay dashboard                            (live page of PCs and jobs)

(Without installing: python gpu_relay.py <same arguments>.)

`send` asks every online, free friend at once. The first to press Enter gets the
job, and the request disappears from everyone else's screen. Logs stream back
live. When the job ends, that friend presses Enter again to send back the files
the job created or changed, into ./my-project/relay-results/<job-id>/.

`batch` hands one line of jobs.txt to each free lab PC, re-runs a job on another
PC if its PC shuts down, stops jobs at --time-limit, collects every result and
shows progress on a dashboard at http://localhost:8766.
"""
import argparse
import fnmatch
import getpass
import hmac
import http.client
import json
import os
import platform
import queue
import re
import secrets
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__version__ = "0.2.1"
PORT = 8765
DASHBOARD_PORT = 8766
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "relay-results", ".DS_Store"}
HOME = Path(os.environ.get("GPU_RELAY_HOME", Path.home() / ".gpu-relay"))
FRIENDS = HOME / "friends.json"
TICKET_SECONDS = 60  # time the sender has to start uploading after approval
HEARTBEAT = 20       # seconds of silence before the worker sends a keep-alive byte
LOST_AFTER = float(os.environ.get("GPU_RELAY_LOST_AFTER", 300))  # unreachable this long = re-run elsewhere
MAX_TRIES = 3        # a batch job is given up after failing to finish on this many PCs
JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}")
MARK = b"__RELAY__ "
# Job priority -> (Windows priority class, Unix nice value). Raising priority above normal on
# macOS/Linux needs root, so there "above-normal" runs at normal.
PRIORITY = {"low": ("BELOW_NORMAL_PRIORITY_CLASS", 10), "normal": (None, 0),
            "above-normal": ("ABOVE_NORMAL_PRIORITY_CLASS", 0)}


def mb(n):
    return f"{n / 1e6:.1f} MB"


def safe_extract(zip_path, dest):
    """Extract a zip, refusing entries that would land outside dest."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError(f"unsafe path in zip: {name}")
        z.extractall(dest)


def zip_project(project, dest):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in project.rglob("*"):
            rel = p.relative_to(project)
            if p.is_file() and not SKIP.intersection(rel.parts):
                z.write(p, rel.as_posix())
    return dest.stat().st_size


# ================================================================ worker side

console = queue.Queue()
console_closed = threading.Event()
prompt_lock = threading.Lock()


def read_console():
    for line in sys.stdin:
        console.put(line)
    console_closed.set()


def ask(question, timeout, withdrawn):
    """Ask the GPU owner. Returns 'yes', 'no', 'timeout' or 'withdrawn'."""
    with prompt_lock:
        while not console.empty():  # ignore keys pressed before this question appeared
            console.get_nowait()
        if console_closed.is_set():
            return "no"
        limit = f" (no answer in {timeout}s = no)" if timeout else ""
        print(f">>> {question}\n    Enter = yes, n + Enter = no{limit}", flush=True)
        deadline = timeout and time.time() + timeout
        while True:
            try:
                return "no" if console.get(timeout=1).strip().lower() in ("n", "no") else "yes"
            except queue.Empty:
                if withdrawn():
                    print("    Request withdrawn (someone else took it, or the sender cancelled).", flush=True)
                    return "withdrawn"
                if deadline and time.time() > deadline:
                    print("    No answer, so declined.", flush=True)
                    return "timeout"


def disconnected(sock):
    """True once the other side has closed an otherwise idle connection."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


def snapshot(folder):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*") if p.is_file()}


def cpu_name():
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def gpu_name():
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            return out.replace("\n", "; ")
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return f"{cpu_name()} GPU (PyTorch device 'mps')"
    return f"no GPU, {cpu_name()}, {os.cpu_count()} threads"


def kill_tree(proc):
    """Stop a job and everything it started (e.g. DataLoader workers)."""
    if not proc or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def keep_awake(on):
    """While a job runs, stop Windows from sleeping (it would pause the job)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


class Job:
    def __init__(self, job_id, folder, command, limit):
        self.id, self.dir, self.command, self.limit = job_id, folder, command, limit
        self.log = folder.parent / f"{job_id}.log"
        self.log.touch()
        self.started, self.ended = time.time(), None
        self.proc = self.code = self.result = None
        self.timed_out = self.cancelled = False
        self.changed = []
        self.done = threading.Event()

    def pack(self):
        zip_path = self.dir.parent / f"{self.id}.result.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in self.changed:
                z.write(p, p.relative_to(self.dir).as_posix())
        self.result = zip_path


def run_job(job, lab, priority):
    """Run job.command in job.dir with output to job.log. Frees the worker when the command ends."""
    try:
        before = snapshot(job.dir)
        command = job.command
        # "python ..." uses the same Python as this worker (the one with PyTorch installed).
        if command == "python" or command.startswith("python "):
            command = f'"{sys.executable}"' + command[len("python"):]
        kw = {}
        win_class, nice = PRIORITY[priority]
        if os.name == "nt":
            args = command
            if win_class:
                kw["creationflags"] = getattr(subprocess, win_class)
        else:
            args = shlex.split(command)
            kw["start_new_session"] = True  # so kill_tree can stop the whole job
            if nice:
                kw["preexec_fn"] = lambda: os.nice(nice)
        keep_awake(True)
        with open(job.log, "ab") as log:
            try:
                job.proc = subprocess.Popen(args, cwd=job.dir, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUNBUFFERED="1"), **kw)
            except OSError as e:
                log.write(f"Could not start the command: {e}\n".encode())
                job.code = 127
            else:
                try:
                    job.code = job.proc.wait(timeout=job.limit or None)
                except subprocess.TimeoutExpired:
                    job.timed_out = True
                    kill_tree(job.proc)
                    job.code = job.proc.wait()
                    log.write(f"\n[gpu-relay] Time limit of {job.limit / 60:g} min reached, job stopped. "
                              f"Files saved so far are sent back.\n".encode())
        job.changed = [p for p, st in snapshot(job.dir).items() if before.get(p) != st]
        if lab and job.changed:
            job.pack()
    finally:
        keep_awake(False)
        job.ended = time.time()
        job.done.set()
        Worker.slot.release()
        print(f"    Job {job.id} ended (exit {job.code}{', time limit' if job.timed_out else ''}). Waiting for jobs.",
              flush=True)


class Worker(BaseHTTPRequestHandler):
    name = token = gpu = ""
    jobs_dir = Path()
    approve_timeout = 180
    lab = False
    priority = "normal"
    slot = threading.Lock()  # held while a job is running
    tickets = {}  # ticket -> (command, expiry time)
    jobs = {}     # job id -> Job

    def log_message(self, *args):
        pass  # keep the owner's console readable

    def reply(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorised(self):
        if hmac.compare_digest(self.headers.get("X-Token", ""), self.token):
            return True
        self.reply(401, {"error": "wrong token"})
        return False

    def busy(self):
        now = time.time()
        for t, (_, expiry) in list(self.tickets.items()):
            if expiry < now:
                self.tickets.pop(t, None)
        return self.slot.locked() or bool(self.tickets)

    def route(self):
        url = urllib.parse.urlsplit(self.path)
        return url.path, {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}

    def do_GET(self):
        if not self.authorised():
            return
        path, query = self.route()
        if path == "/status":
            return self.reply(200, {"name": self.name, "gpu": self.gpu, "os": platform.system(), "busy": self.busy(),
                                    "lab": self.lab, "priority": self.priority, "version": __version__})
        job = self.jobs.get(path.split("/")[-1])
        if path.startswith("/job/") and job:
            return self.job_progress(job, int(query.get("since", 0)))
        if not path.startswith("/result/") or not job or not job.result:
            return self.reply(404, {"error": "no such job or result"})
        self.send_response(200)
        self.send_header("Content-Length", str(job.result.stat().st_size))
        self.end_headers()
        with open(job.result, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        if not self.authorised():
            return
        path, query = self.route()
        if path == "/ask":
            return self.handle_ask()
        if path == "/run":
            return self.handle_run()
        if path == "/submit":
            return self.handle_submit(query)
        job = self.jobs.get(path.split("/")[-1])
        if path.startswith("/cancel/") and job:
            job.cancelled = True
            kill_tree(job.proc)
            return self.reply(200, {"cancelled": job.id})
        self.reply(404, {"error": "unknown path"})

    # ---- starting a job

    def start(self, command, limit):
        """Receive the uploaded project and start the command. Caller holds the slot."""
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        folder = self.jobs_dir / job_id
        upload = self.jobs_dir / f"{job_id}.upload.zip"
        try:
            folder.mkdir(parents=True)
            remaining = int(self.headers.get("Content-Length", 0))
            with open(upload, "wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise ConnectionError("upload cut off")
                    f.write(chunk)
                    remaining -= len(chunk)
            safe_extract(upload, folder)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            self.slot.release()
            shutil.rmtree(folder, ignore_errors=True)
            try:
                self.reply(400, {"error": f"bad upload: {e}"})
            except OSError:
                pass
            return None
        finally:
            upload.unlink(missing_ok=True)
        self.forget_old_jobs()
        job = self.jobs[job_id] = Job(job_id, folder, command, limit)
        threading.Thread(target=run_job, args=(job, self.lab, self.priority), daemon=True).start()
        print(f"\n    Running job {job_id}: {command}", flush=True)
        return job

    def forget_old_jobs(self, keep=2):
        """Lab PCs keep only the newest finished jobs on disk (the sender has already collected them)."""
        if not self.lab:
            return
        finished = sorted(j for j, job in self.jobs.items() if job.done.is_set())
        for j in finished[:-keep]:
            job = self.jobs.pop(j)
            shutil.rmtree(job.dir, ignore_errors=True)
            for p in (job.log, job.result):
                if p:
                    p.unlink(missing_ok=True)

    def handle_submit(self, query):
        """Lab mode: start a job straight away, no questions. The sender polls /job/<id>."""
        if not self.lab:
            return self.reply(403, {"error": "this computer asks before running jobs (use send)"})
        if self.tickets or not self.slot.acquire(blocking=False):
            return self.reply(409, {"error": "busy"})
        job = self.start(query.get("command", ""), float(query.get("limit", 0)))
        if job:
            self.reply(200, {"job_id": job.id})

    def job_progress(self, job, since):
        finished = job.done.is_set()  # read before the log so no output is missed
        with open(job.log, "rb") as f:
            f.seek(since)
            data = f.read(1 << 18)
            size = f.seek(0, 2)
        done = finished and since + len(data) >= size
        self.reply(200, {"state": "done" if done else "running", "code": job.code, "timed_out": job.timed_out,
                         "cancelled": job.cancelled, "elapsed": (job.ended or time.time()) - job.started,
                         "log": data.decode("utf-8", "replace"), "next": since + len(data),
                         "result": bool(job.result), "files": len(job.changed)})

    # ---- interactive: ask the owner, then stream the job live

    def handle_ask(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.busy():
            return self.reply(409, {"error": "busy"})
        print(f"\n=== Job request from {req['sender']} ({self.client_address[0]})"
              f"\n    project: {req['project']} ({mb(req['size'])})\n    command: {req['command']}", flush=True)
        answer = "yes" if self.lab else ask("Run this job on your GPU?", self.approve_timeout,
                                             lambda: disconnected(self.connection))
        if answer == "withdrawn":
            return
        if answer != "yes":
            return self.reply(403, {"error": "declined" if answer == "no" else "no answer"})
        if self.busy():  # another sender got here first while we were asking
            return self.reply(409, {"error": "busy"})
        ticket = secrets.token_urlsafe(12)
        self.tickets[ticket] = (req["command"], time.time() + TICKET_SECONDS)
        try:
            self.reply(200, {"ticket": ticket})
        except ConnectionError:
            self.tickets.pop(ticket, None)

    def handle_run(self):
        command, expiry = self.tickets.pop(self.headers.get("X-Ticket", ""), ("", 0))
        if expiry < time.time():
            return self.reply(403, {"error": "no valid approval for this upload"})
        if not self.slot.acquire(blocking=False):
            return self.reply(409, {"error": "busy"})
        job = self.start(command, 0)
        if not job:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        sender_alive = self.stream_log(job)
        size = sum(p.stat().st_size for p in job.changed)
        if job.changed and sender_alive and not self.lab and ask(
                f"Job finished (exit code {job.code}). Send {len(job.changed)} result file(s), {mb(size)}, back?",
                None, lambda: disconnected(self.connection)) == "yes":
            job.pack()
        sent = "sent" if job.result else "withheld"
        print(f"    Results {sent}.", flush=True)
        try:
            self.wfile.write(b"\n" + MARK + f"{job.id} {job.code} {sent} {len(job.changed)}\n".encode())
        except ConnectionError:
            pass

    def stream_log(self, job):
        """Copy the job's output to the sender (and this console) until it ends. False if the sender left."""
        quiet_since = time.time()
        with open(job.log, "rb") as f:
            while True:
                ended = job.done.is_set()
                data = f.read(1 << 16)
                try:
                    if data:
                        if not self.lab:
                            sys.stdout.buffer.write(data)
                            sys.stdout.flush()
                        self.wfile.write(data)
                        self.wfile.flush()
                        quiet_since = time.time()
                    elif ended:
                        return True
                    else:
                        if disconnected(self.connection):
                            raise ConnectionError
                        if time.time() - quiet_since > HEARTBEAT:
                            self.wfile.write(b"\0")  # keep-alive; the sender drops it
                            self.wfile.flush()
                            quiet_since = time.time()
                        time.sleep(0.2)
                except ConnectionError:
                    print("\n    Sender disconnected; stopping the job.", flush=True)
                    kill_tree(job.proc)
                    job.done.wait()
                    return False


def worker(args):
    if args.log:
        sys.stdout = sys.stderr = open(args.log, "a", buffering=1, encoding="utf-8", errors="replace")
    jobs = Path(args.jobs).resolve()
    jobs.mkdir(parents=True, exist_ok=True)
    if args.lab:  # leftovers from before a restart can't be collected any more
        for p in jobs.iterdir():
            if JOB_ID.match(p.name):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    token_file = jobs.parent / "relay-token.txt"
    token = os.environ.get("GPU_RELAY_TOKEN")
    if not token:
        if not token_file.exists():
            token_file.write_text(secrets.token_urlsafe(16))
        token = token_file.read_text().strip()
    Worker.name, Worker.token, Worker.gpu, Worker.lab = args.name, token, gpu_name(), args.lab
    Worker.jobs_dir, Worker.approve_timeout, Worker.priority = jobs, args.approve_timeout, args.priority
    if not args.lab:
        threading.Thread(target=read_console, daemon=True).start()
    mode = (f"Lab mode: jobs from anyone with the token run automatically, at {args.priority} priority."
            if args.lab else "Each job and each result send needs your Enter.")
    print(f"{time.strftime('%Y-%m-%d %H:%M')} gpu-relay {__version__} worker '{args.name}' on port {args.port}"
          f"\n  {mode}\n  GPU:   {Worker.gpu}\n  token: {token}  (give this to the sender once)"
          f"\n  jobs:  {jobs}\nWaiting for jobs. Ctrl+C to stop.", flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Worker).serve_forever()
    except KeyboardInterrupt:
        for job in Worker.jobs.values():
            kill_tree(job.proc)
        print("Stopped.")


# ================================================================ sender side

def load_friends():
    return json.loads(FRIENDS.read_text()) if FRIENDS.exists() else {}


def save_friends(friends):
    HOME.mkdir(parents=True, exist_ok=True)
    FRIENDS.write_text(json.dumps(friends, indent=2))
    try:
        FRIENDS.chmod(0o600)  # holds tokens
    except OSError:
        pass


def call(friend, method, path, body=None, headers=None, timeout=None):
    conn = http.client.HTTPConnection(friend["host"], friend["port"], timeout=timeout)
    conn.request(method, path, body=body, headers={"X-Token": friend["token"], **(headers or {})})
    return conn, conn.getresponse()


def call_json(friend, method, path, timeout=30, **kw):
    conn, r = call(friend, method, path, timeout=timeout, **kw)
    try:
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


def status(friend):
    try:
        code, data = call_json(friend, "GET", "/status", timeout=4)
        if code == 401:
            return "wrong token", data
        return ("busy" if data["busy"] else "online"), data
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        return "offline", {}


def check_all(names, friends):
    with ThreadPoolExecutor(max(1, min(len(names), 64))) as pool:
        return dict(zip(names, pool.map(lambda n: status(friends[n]), names)))


def pick(patterns, friends):
    """Friend names matching the given names or wildcards like 'lab1-*' (all friends if none given)."""
    if not friends:
        sys.exit("No friends yet. Add one: gpu-relay add <name> <address> <token>")
    if not patterns:
        return list(friends)
    missing = [p for p in patterns if not any(fnmatch.fnmatch(n, p) for n in friends)]
    if missing:
        sys.exit(f"Unknown friend(s): {', '.join(missing)}. See: gpu-relay friends")
    return [n for n in friends if any(fnmatch.fnmatch(n, p) for p in patterns)]


def describe(data):
    return f"{data.get('os', '')}, {data['gpu']}" + (" [lab]" if data.get("lab") else "") if "gpu" in data else ""


def cmd_friends(args):
    friends = load_friends()
    names = pick(args.to, friends)
    for name, (state, data) in check_all(names, friends).items():
        print(f"  {name:<14} {state:<11} {describe(data) or friends[name]['host']}")


def cmd_add(args):
    friends = load_friends()
    friends[args.name] = {"host": args.host, "port": args.port, "token": args.token}
    save_friends(friends)
    state, data = status(friends[args.name])
    print(f"Saved {args.name}. Right now: {state}" + (f" ({data['gpu']})" if "gpu" in data else ""))


def cmd_add_many(args):
    friends = load_friends()
    for name in args.names:
        friends[name] = {"host": name, "port": args.port, "token": args.token}
    save_friends(friends)
    states = check_all(args.names, friends)
    print(f"Saved {len(args.names)} PC(s). Right now:")
    for name, (state, data) in states.items():
        print(f"  {name:<14} {state:<11} {describe(data)}")


def cmd_remove(args):
    friends = load_friends()
    names = pick([args.name], friends)
    for n in names:
        friends.pop(n)
    save_friends(friends)
    print(f"Removed {', '.join(names)}.")


def broadcast(names, friends, request):
    """Ask everyone at once; return (name, ticket) of the first to accept, or None."""
    answers, conns, done = queue.Queue(), {}, threading.Event()

    def ask_one(name):
        try:
            conn = http.client.HTTPConnection(friends[name]["host"], friends[name]["port"])
            conn.connect()
            conns[name] = conn
            if done.is_set():  # someone already accepted
                return conn.close()
            conn.request("POST", "/ask", body=json.dumps(request),
                         headers={"X-Token": friends[name]["token"], "Content-Type": "application/json"})
            r = conn.getresponse()
            answers.put((name, r.status, json.loads(r.read())))
        except (OSError, ValueError, http.client.HTTPException) as e:
            answers.put((name, None, {"error": f"connection lost ({e.__class__.__name__})"}))

    for name in names:
        threading.Thread(target=ask_one, args=(name,), daemon=True).start()
    winner = None
    for _ in names:
        name, code, data = answers.get()
        if code == 200:
            winner = (name, data["ticket"])
            break
        print(f"  {name}: {data.get('error', code)}")
    done.set()
    for name, conn in list(conns.items()):  # withdraw the request from everyone else
        if not winner or name != winner[0]:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
    return winner


def stream(r, out):
    """Copy a live job's output to out(bytes). Returns the worker's final status fields, or None."""
    status_fields, buf = None, b""
    while True:
        chunk = r.read1(1 << 16)
        if not chunk:
            return status_fields
        buf += chunk.replace(b"\0", b"")  # drop keep-alive bytes
        *lines, buf = buf.split(b"\n")
        for line in lines:
            if line.startswith(MARK):
                status_fields = line.decode().split()[1:]
            else:
                out(line + b"\n")
        if buf and not MARK.startswith(buf[:len(MARK)]):  # show progress bars (\r, no \n) right away
            out(buf)
            buf = b""


def cmd_send(args):
    friends = load_friends()
    names = pick(args.to, friends)
    states = check_all(names, friends)
    free = [n for n in names if states[n][0] == "online"]
    for n in names:
        if n not in free:
            print(f"  {n}: {states[n][0]}")
    if not free:
        sys.exit("Nobody is online and free right now.")

    project = Path(args.project).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        upload = Path(tmp) / "job.zip"
        size = zip_project(project, upload)
        print(f"Asking {', '.join(free)} to run \"{args.command}\" on {project.name} ({mb(size)})...", flush=True)
        winner = broadcast(free, friends, {"sender": f"{getpass.getuser()}@{socket.gethostname()}",
                                           "project": project.name, "size": size, "command": args.command})
        if not winner:
            sys.exit("Nobody accepted the job.")
        name, ticket = winner
        friend = friends[name]
        print(f"{name} accepted. Uploading and running on {states[name][1].get('gpu', 'their GPU')}...\n", flush=True)

        def show(data):
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

        with open(upload, "rb") as f:
            try:
                conn, r = call(friend, "POST", "/run", body=f, timeout=HEARTBEAT * 6, headers={
                    "X-Ticket": ticket, "Content-Length": str(size), "Content-Type": "application/zip"})
                if r.status != 200:
                    sys.exit(f"{name}'s laptop refused the upload: {r.read().decode(errors='replace')}")
                status_line = stream(r, show)
                conn.close()
            except (OSError, http.client.HTTPException) as e:
                sys.exit(f"\nLost the connection to {name} ({e.__class__.__name__}).")

        if not status_line:
            sys.exit("Connection ended before the job finished.")
        job_id, code, state, count = status_line
        if state != "sent":
            print(f"\nJob {job_id} on {name} exited with code {code}; results not sent ({count} changed file(s)).")
            sys.exit(int(code))
        result = Path(tmp) / "result.zip"
        conn, r = call(friend, "GET", f"/result/{job_id}")
        with open(result, "wb") as out:
            shutil.copyfileobj(r, out)
        conn.close()
        dest = project / "relay-results" / job_id
        safe_extract(result, dest)
        print(f"\nJob {job_id} on {name} exited with code {code}. {count} result file(s) saved to {dest}")
        sys.exit(int(code))


# ---------------------------------------------------------------- batch + dashboard

FINAL = ("done", "failed", "time limit", "cancelled", "gave up")


def last_line(text):
    lines = [s for s in re.split(r"[\r\n]+", text) if s.strip()]
    return lines[-1].strip()[-200:] if lines else ""


class Batch:
    """Hands jobs to free lab PCs, follows them, re-runs lost ones, collects results.
    State is saved to <folder>/batch.json so `batch --resume <folder>` can pick up again."""

    def __init__(self, friends, names, folder=None, jobs=(), limit=0, zip_path=None):
        self.friends, self.names, self.dir, self.limit, self.zip = friends, names, folder, limit, zip_path
        self.jobs = list(jobs)
        self.pcs = {n: {"state": "checking", "info": "", "seen": None, "lab": False} for n in names}
        self.lock = threading.Lock()

    def save(self):
        if self.dir:
            with self.lock:
                (self.dir / "batch.json.tmp").write_text(
                    json.dumps({"names": self.names, "limit": self.limit, "jobs": self.jobs}, indent=1))
                os.replace(self.dir / "batch.json.tmp", self.dir / "batch.json")

    def note(self, job, line):
        with open(self.dir / f"job{job['n']:03}.log", "a", encoding="utf-8") as f:
            f.write(f"\n[gpu-relay] {line}\n")

    def requeue(self, job, why, count=True):
        with self.lock:
            job["tries"] += count
            again = job["tries"] < MAX_TRIES
            self.note(job, f"{why} on {job['pc']}; " + ("will run again on another PC" if again else "giving up"))
            job.update(status="pending" if again else "gave up", note=f"{why} on {job['pc']}",
                       pc=None, job_id=None, offset=0, elapsed=0, last="")
        self.save()

    def follow(self, job):
        try:
            self._follow(job)
        except Exception as e:  # never leave a job stuck as "running"
            self.requeue(job, f"unexpected error ({e!r})")

    def _follow(self, job):
        """Upload a job to its PC (unless already running there), follow its log, fetch its results."""
        friend = self.friends[job["pc"]]
        if not job["job_id"]:
            try:
                with open(self.zip, "rb") as f:
                    code, data = call_json(friend, "POST", "/submit?" + urllib.parse.urlencode(
                        {"command": job["command"], "limit": self.limit}), body=f, timeout=120,
                        headers={"Content-Length": str(self.zip.stat().st_size), "Content-Type": "application/zip"})
            except (OSError, ValueError, http.client.HTTPException) as e:
                return self.requeue(job, f"upload failed ({e.__class__.__name__})")
            if code != 200:
                return self.requeue(job, data.get("error", f"HTTP {code}"), count=code != 409)
            with self.lock:
                job.update(job_id=data["job_id"], status="running", started=time.time())
            self.save()
        unreachable = None
        with open(self.dir / f"job{job['n']:03}.log", "a", encoding="utf-8") as log:
            while True:
                try:
                    code, data = call_json(friend, "GET", f"/job/{job['job_id']}?since={job['offset']}")
                except (OSError, ValueError, http.client.HTTPException):
                    unreachable = unreachable or time.time()
                    if time.time() - unreachable > LOST_AFTER:
                        return self.requeue(job, "PC went offline")
                    time.sleep(5)
                    continue
                unreachable = None
                if code != 200:
                    return self.requeue(job, "PC restarted" if code == 404 else f"HTTP {code}")
                log.write(data["log"])
                log.flush()
                with self.lock:
                    job["tail"] = (job.get("tail", "") + data["log"])[-4000:]
                    job.update(offset=data["next"], elapsed=data["elapsed"], last=last_line(job["tail"]))
                if data["state"] == "done":
                    break
                time.sleep(2)
        dest = self.dir / f"job{job['n']:03}-{job['pc']}"
        if data["result"]:
            try:
                conn, r = call(friend, "GET", f"/result/{job['job_id']}", timeout=120)
                with open(self.dir / f"job{job['n']:03}.result.zip", "wb") as out:
                    shutil.copyfileobj(r, out)
                conn.close()
                safe_extract(self.dir / f"job{job['n']:03}.result.zip", dest)
                (self.dir / f"job{job['n']:03}.result.zip").unlink()
            except (OSError, ValueError, zipfile.BadZipFile, http.client.HTTPException) as e:
                return self.requeue(job, f"could not fetch results ({e.__class__.__name__})")
        final = ("cancelled" if data["cancelled"] else "time limit" if data["timed_out"]
                 else "done" if data["code"] == 0 else "failed")
        with self.lock:
            job.update(status=final, code=data["code"], files=data["files"] if data["result"] else 0,
                       results=str(dest) if data["result"] else "")
        self.save()

    def refresh(self):
        states = check_all(self.names, self.friends)
        with self.lock:
            for n, (state, data) in states.items():
                pc = self.pcs[n]
                pc.update(state=state, lab=data.get("lab", False))
                if state != "offline":
                    pc.update(seen=time.time(), info=describe(data))
            taken = {j["pc"] for j in self.jobs if j["status"] in ("starting", "running")}
            free = [n for n in self.names if states[n][0] == "online" and states[n][1].get("lab") and n not in taken]
            for job in self.jobs:
                if job["status"] == "pending" and free:
                    job.update(status="starting", pc=free.pop(0))
                    threading.Thread(target=self.follow, args=(job,), daemon=True).start()

    def summary(self):
        with self.lock:
            counts = {}
            for j in self.jobs:
                counts[j["status"]] = counts.get(j["status"], 0) + 1
            online = sum(p["state"] in ("online", "busy") for p in self.pcs.values())
        jobs = ", ".join(f"{v} {k}" for k, v in counts.items())
        return f"{online}/{len(self.pcs)} PCs online" + (f" | jobs: {jobs}" if jobs else "")

    def cancel_running(self):
        running = [j for j in self.jobs if j["status"] == "running"]
        for j in running:
            try:
                call_json(self.friends[j["pc"]], "POST", f"/cancel/{j['job_id']}", timeout=5)
            except (OSError, ValueError, http.client.HTTPException):
                pass
            j["status"] = "cancelled"
        self.save()
        return len(running)

    def snapshot(self):
        with self.lock:
            return {"now": time.time(), "limit": self.limit, "folder": str(self.dir or ""),
                    "pcs": [{"name": n, **p, "job": next((j["n"] for j in self.jobs if j["pc"] == n and
                                                          j["status"] in ("starting", "running")), None)}
                            for n, p in self.pcs.items()],
                    "jobs": [{k: v for k, v in j.items() if k != "tail"} for j in self.jobs]}

    def log_tail(self, n):
        try:
            with open(self.dir / f"job{n:03}.log", "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 20000))
                return f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            return ""


def serve_dashboard(batch, port):
    class Page(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/state":
                body, kind = json.dumps(batch.snapshot()).encode(), "application/json"
            elif self.path.startswith("/log/") and self.path[5:].isdigit():
                body, kind = batch.log_tail(int(self.path[5:])).encode(), "text/plain; charset=utf-8"
            else:
                body, kind = DASHBOARD_HTML.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Page)
    except OSError:
        print(f"Dashboard port {port} is in use (is another dashboard open?). Use --dashboard-port.")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"Dashboard: {url}", flush=True)
    if not os.environ.get("GPU_RELAY_NO_BROWSER"):
        webbrowser.open(url)


def run_batch(batch, port):
    serve_dashboard(batch, port)
    for job in batch.jobs:  # after --resume: keep following jobs still running on their PCs
        if job["status"] == "running":
            threading.Thread(target=batch.follow, args=(job,), daemon=True).start()
    shown = None
    try:
        while True:
            batch.refresh()
            batch.save()
            line = batch.summary()
            if line != shown:
                print(f"{time.strftime('%H:%M:%S')}  {line}", flush=True)
                shown = line
            if batch.jobs and all(j["status"] in FINAL for j in batch.jobs):
                break
            time.sleep(5)
    except KeyboardInterrupt:
        n = batch.cancel_running()
        sys.exit(f"\nStopped; cancelled {n} running job(s). Re-run the unfinished ones later with:\n"
                 f"  gpu-relay batch --resume \"{batch.dir}\"" if batch.dir else "")
    bad = [j for j in batch.jobs if j["status"] != "done"]
    print(f"\nBatch finished: {len(batch.jobs) - len(bad)} done, {len(bad)} not. Results and logs in {batch.dir}")
    for j in bad:
        print(f"  job {j['n']} ({j['command']}): {j['status']}" + (f", {j['note']}" if j.get("note") else ""))
    time.sleep(3)  # let the dashboard show the final state
    sys.exit(1 if bad else 0)


def cmd_batch(args):
    friends = load_friends()
    if args.resume:
        folder = Path(args.resume).resolve()
        saved = json.loads((folder / "batch.json").read_text())
        for j in saved["jobs"]:
            if j["status"] in ("starting", "cancelled"):
                j.update(status="pending", pc=None, job_id=None, offset=0)
        names = [n for n in saved["names"] if n in friends]
        batch = Batch(friends, names, folder, saved["jobs"], saved["limit"], folder / "project.zip")
        return run_batch(batch, args.dashboard_port)
    if not args.project or not args.jobs_file:
        sys.exit("Usage: gpu-relay batch <project> <jobs.txt> [--to NAME...]   or   gpu-relay batch --resume <folder>")
    commands = [s.strip() for s in Path(args.jobs_file).read_text().splitlines()
                if s.strip() and not s.strip().startswith("#")]
    if not commands:
        sys.exit(f"No commands in {args.jobs_file} (one per line).")
    names = pick(args.to, friends)
    project = Path(args.project).resolve()
    folder = project / "relay-results" / time.strftime("batch-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    size = zip_project(project, folder / "project.zip")
    print(f"{len(commands)} job(s) from {args.jobs_file}, project {project.name} ({mb(size)}), "
          f"time limit {args.time_limit:g} min, on {len(names)} PC(s).\nResults: {folder}", flush=True)
    jobs = [{"n": i, "command": c, "status": "pending", "pc": None, "job_id": None, "tries": 0, "offset": 0,
             "elapsed": 0, "last": "", "note": ""} for i, c in enumerate(commands, 1)]
    batch = Batch(friends, names, folder, jobs, args.time_limit * 60, folder / "project.zip")
    run_batch(batch, args.dashboard_port)


def cmd_dashboard(args):
    friends = load_friends()
    batch = Batch(friends, pick(args.to, friends))
    serve_dashboard(batch, args.dashboard_port)
    try:
        while True:
            batch.refresh()
            time.sleep(5)
    except KeyboardInterrupt:
        pass


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>gpu-relay dashboard</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1b1f24;--muted:#667085;--line:#e4e7ec;--ok:#12a150;--run:#2f6fed;
--off:#98a2b3;--bad:#d92d20;--warn:#dc8a00;--bar:#eef1f5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--card:#171b21;--text:#e6e9ee;--muted:#98a2b3;--line:#272d36;
--bar:#232a33}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 10px}.muted{color:var(--muted)}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:110px}
.stat b{display:block;font-size:22px}
.pcs{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px}
.pc{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--off);border-radius:8px;padding:8px 10px}
.pc.online{border-left-color:var(--ok)}.pc.busy{border-left-color:var(--run)}.pc.wrong{border-left-color:var(--bad)}
.pc .n{font-weight:600}.pc .s{font-size:12px}.pc .i{font-size:11px;color:var(--muted);overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.tablewrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:760px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{font-size:12px;color:var(--muted);font-weight:600}
tr.job{cursor:pointer}tr.job:hover{background:var(--bar)}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;color:#fff;background:var(--off)}
.tag.running,.tag.starting{background:var(--run)}.tag.done{background:var(--ok)}.tag.failed,.tag.gave{background:var(--bad)}
.tag.time{background:var(--warn)}
.bar{height:6px;background:var(--bar);border-radius:3px;overflow:hidden;min-width:90px;margin-top:4px}
.bar i{display:block;height:100%;background:var(--run)}.bar i.t{background:var(--warn)}
code{font:12px ui-monospace,Menlo,Consolas,monospace}.last{font:12px ui-monospace,Menlo,Consolas,monospace;
color:var(--muted);max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;max-height:420px;overflow:auto;
font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
</style></head><body><main>
<h1>gpu-relay</h1><div class="muted" id="sub">Connecting...</div>
<div class="stats" id="stats"></div>
<h2>Computers</h2><div class="pcs" id="pcs"></div>
<div id="jobsec"><h2>Jobs</h2><div class="tablewrap"><table><thead><tr><th>#</th><th>Command</th><th>Status</th>
<th>PC</th><th>Time used</th><th>Progress</th><th>Latest output</th></tr></thead><tbody id="jobs"></tbody></table></div>
<div id="logbox" hidden><h2 id="logtitle"></h2><pre id="log"></pre></div></div>
</main><script>
const $=id=>document.getElementById(id), esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const mins=s=>s<60?Math.round(s)+"s":Math.floor(s/60)+"m "+String(Math.round(s%60)).padStart(2,"0")+"s";
let openLog=null;
function ago(t,now){if(!t)return"never seen";const s=now-t;return s<90?"seen just now":"last seen "+mins(s)+" ago"}
async function tick(){
 try{const d=await (await fetch("/state")).json();
  const on=d.pcs.filter(p=>p.state=="online"||p.state=="busy").length, c={};
  d.jobs.forEach(j=>c[j.status]=(c[j.status]||0)+1);
  $("sub").textContent=(d.folder?"Results: "+d.folder+" · ":"")+(d.limit?"time limit "+d.limit/60+" min per job · ":"")+"updated "+new Date().toLocaleTimeString();
  $("stats").innerHTML=[["PCs online",on+" / "+d.pcs.length],...Object.entries(c)].map(([k,v])=>`<div class="stat"><b>${esc(v)}</b>${esc(k)}</div>`).join("");
  $("pcs").innerHTML=d.pcs.map(p=>{const st=p.state=="busy"?"busy":p.state=="online"?"online":p.state=="wrong token"?"wrong":"off";
   const lbl=p.job?"running job "+p.job:p.state=="online"?(p.lab?"online, free":"online (asks first)"):p.state;
   return `<div class="pc ${st}"><div class="n">${esc(p.name)}</div><div class="s">${esc(lbl)}</div><div class="i" title="${esc(p.info)}">${esc(p.state=="offline"?ago(p.seen,d.now):p.info)}</div></div>`}).join("");
  $("jobsec").hidden=!d.jobs.length;
  $("jobs").innerHTML=d.jobs.map(j=>{const pct=(j.last.match(/(\d{1,3}(?:\.\d+)?)\s?%/g)||[]).pop(), p=pct?Math.min(100,parseFloat(pct)):null;
   const t=d.limit?Math.min(100,100*j.elapsed/d.limit):0;
   return `<tr class="job" data-n="${j.n}"><td>${j.n}</td><td><code>${esc(j.command)}</code></td><td><span class="tag ${esc(j.status.split(" ")[0])}">${esc(j.status)}</span>${j.note?`<div class="muted" style="font-size:11px">${esc(j.note)}</div>`:""}</td>
   <td>${esc(j.pc||"")}</td><td>${j.elapsed?mins(j.elapsed):""}${d.limit&&j.elapsed?`<div class="bar"><i class="t" style="width:${t}%"></i></div>`:""}</td>
   <td>${p!==null?p+"%<div class='bar'><i style='width:"+p+"%'></i></div>":""}</td><td><div class="last" title="${esc(j.last)}">${esc(j.last)}</div></td></tr>`}).join("");
  document.querySelectorAll("tr.job").forEach(r=>r.onclick=()=>{openLog=+r.dataset.n;showLog()});
  if(openLog)showLog();
 }catch(e){$("sub").textContent="Dashboard stopped (the gpu-relay command is no longer running)."}
}
async function showLog(){$("logbox").hidden=false;$("logtitle").textContent="Output of job "+openLog;
 const pre=$("log"), stick=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-20;
 const raw=await (await fetch("/log/"+openLog)).text();  // progress bars redraw with \r: keep each line's last state
 pre.textContent=raw.split("\n").map(l=>l.split("\r").filter(s=>s.trim()).pop()||"").join("\n")||"(no output yet)"; if(stick)pre.scrollTop=pre.scrollHeight}
tick();setInterval(tick,2000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(prog="gpu-relay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"gpu-relay {__version__}")
    sub = ap.add_subparsers(dest="mode", required=True)

    w = sub.add_parser("worker", help="share this computer")
    w.add_argument("--name", default=socket.gethostname().lower(), help="name shown to senders")
    w.add_argument("--host", default="0.0.0.0")
    w.add_argument("--port", type=int, default=PORT)
    w.add_argument("--jobs", default=str(HOME / "jobs"), help="where jobs run (default ~/.gpu-relay/jobs)")
    w.add_argument("--approve-timeout", type=int, default=180, help="seconds before an unanswered request counts as no")
    w.add_argument("--lab", action="store_true",
                   help="run jobs automatically, no Enter needed (lab PCs you may use)")
    w.add_argument("--priority", choices=list(PRIORITY), default="normal",
                   help="CPU priority of jobs (default normal; low = students' programs always come first)")
    w.add_argument("--log", help="write messages to this file instead of the screen (background use)")
    w.set_defaults(fn=worker)

    a = sub.add_parser("add", help="save a friend's laptop")
    a.add_argument("name")
    a.add_argument("host", help="Tailscale name or 100.x.y.z address")
    a.add_argument("token")
    a.add_argument("--port", type=int, default=PORT)
    a.set_defaults(fn=cmd_add)

    m = sub.add_parser("add-many", help="save several lab PCs that share one token (name = Tailscale name)")
    m.add_argument("token")
    m.add_argument("names", nargs="+", metavar="NAME")
    m.add_argument("--port", type=int, default=PORT)
    m.set_defaults(fn=cmd_add_many)

    r = sub.add_parser("remove", help="forget a friend (wildcards like 'lab1-*' work)")
    r.add_argument("name")
    r.set_defaults(fn=cmd_remove)

    f = sub.add_parser("friends", help="show who is online")
    f.add_argument("--to", nargs="+", metavar="NAME", help="only these (wildcards like 'lab1-*' work)")
    f.set_defaults(fn=cmd_friends)

    s = sub.add_parser("send", help="run a project on whoever accepts first")
    s.add_argument("project", help="project folder to send")
    s.add_argument("command", help='command to run inside it, e.g. "python train.py"')
    s.add_argument("--to", nargs="+", metavar="NAME", help="only ask these (wildcards like 'lab1-*' work)")
    s.set_defaults(fn=cmd_send)

    b = sub.add_parser("batch", help="run many jobs on lab PCs, one per free PC, with a live dashboard")
    b.add_argument("project", nargs="?", help="project folder to send")
    b.add_argument("jobs_file", nargs="?", help="text file with one command per line")
    b.add_argument("--to", nargs="+", metavar="NAME", help="only use these PCs (wildcards like 'lab1-*' work)")
    b.add_argument("--time-limit", type=float, default=90,
                   help="minutes a job may run before it is stopped and its files sent back (default 90)")
    b.add_argument("--resume", metavar="FOLDER", help="pick up a batch again (its relay-results/batch-... folder)")
    b.add_argument("--dashboard-port", type=int, default=DASHBOARD_PORT)
    b.set_defaults(fn=cmd_batch)

    d = sub.add_parser("dashboard", help="live page showing which computers are online")
    d.add_argument("--to", nargs="+", metavar="NAME", help="only show these (wildcards like 'lab1-*' work)")
    d.add_argument("--dashboard-port", type=int, default=DASHBOARD_PORT)
    d.set_defaults(fn=cmd_dashboard)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
