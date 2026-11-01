"""GPU relay: run a project on a friend's GPU laptop, with their approval.

Works on Windows, macOS and Linux (Python 3.9+, standard library only).

Friend's laptop:  gpu-relay worker --name ravi
Your computer:    gpu-relay add ravi <address> <token>     (once per friend)
                  gpu-relay friends                         (who is online)
                  gpu-relay send ./my-project "python train.py"

(Without installing: python gpu_relay.py <same arguments>.)

`send` asks every online, free friend at once. The first to press Enter gets the
job, and the request disappears from everyone else's screen. Logs stream back
live. When the job ends, that friend presses Enter again to send back the files
the job created or changed, into ./my-project/relay-results/<job-id>/.
"""
import argparse
import getpass
import hmac
import http.client
import json
import os
import platform
import queue
import secrets
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__version__ = "0.1.0"
PORT = 8765
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "relay-results", ".DS_Store"}
HOME = Path(os.environ.get("GPU_RELAY_HOME", Path.home() / ".gpu-relay"))
FRIENDS = HOME / "friends.json"
TICKET_SECONDS = 60  # time the sender has to start uploading after approval


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


def gpu_name():
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            return out.replace("\n", "; ")
    if sys.platform == "darwin" and platform.machine() == "arm64":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        return f"{chip.strip()} GPU (PyTorch device 'mps')"
    return "no GPU found (CPU only)"


class Worker(BaseHTTPRequestHandler):
    name = token = gpu = ""
    jobs_dir = Path()
    approve_timeout = 180
    running = threading.Lock()
    tickets = {}  # ticket -> (command, expiry time)
    results = {}  # job id -> result zip, only for approved sends

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
        return self.running.locked() or bool(self.tickets)

    def do_GET(self):
        if not self.authorised():
            return
        if self.path == "/status":
            return self.reply(200, {"name": self.name, "gpu": self.gpu, "os": platform.system(), "busy": self.busy()})
        zip_path = self.results.get(self.path.removeprefix("/result/"))
        if not self.path.startswith("/result/") or not zip_path:
            return self.reply(404, {"error": "no such result"})
        self.send_response(200)
        self.send_header("Content-Length", str(zip_path.stat().st_size))
        self.end_headers()
        with open(zip_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_POST(self):
        if not self.authorised():
            return
        if self.path == "/ask":
            return self.handle_ask()
        if self.path == "/run":
            return self.handle_run()
        self.reply(404, {"error": "unknown path"})

    def handle_ask(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.busy():
            return self.reply(409, {"error": "busy"})
        print(f"\n=== Job request from {req['sender']} ({self.client_address[0]})"
              f"\n    project: {req['project']} ({mb(req['size'])})\n    command: {req['command']}", flush=True)
        answer = ask("Run this job on your GPU?", self.approve_timeout, lambda: disconnected(self.connection))
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
        with self.running:
            self.run_job(command)

    def run_job(self, command):
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True)
        upload = self.jobs_dir / f"{job_id}.upload.zip"
        remaining = int(self.headers.get("Content-Length", 0))
        with open(upload, "wb") as f:
            while remaining:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        safe_extract(upload, job_dir)
        upload.unlink()
        before = snapshot(job_dir)
        print(f"    Running job {job_id}: {command}\n", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        # "python ..." uses the same Python as this worker (the one with GPU PyTorch installed).
        if command == "python" or command.startswith("python "):
            command = f'"{sys.executable}"' + command[len("python"):]
        args = command if os.name == "nt" else shlex.split(command)
        try:
            proc = subprocess.Popen(args, cwd=job_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except OSError as e:
            self.wfile.write(f"Could not start the command: {e}\n__RELAY__ {job_id} 127 withheld 0\n".encode())
            return
        sender_alive = True
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            if sender_alive:
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except ConnectionError:
                    sender_alive = False
                    print("\n    Sender disconnected; stopping the job.", flush=True)
                    proc.kill()
        code = proc.wait()

        changed = [p for p, st in snapshot(job_dir).items() if before.get(p) != st]
        size = sum(p.stat().st_size for p in changed)
        sent = False
        if changed and sender_alive and ask(
                f"Job finished (exit code {code}). Send {len(changed)} result file(s), {mb(size)}, back?",
                None, lambda: disconnected(self.connection)) == "yes":
            zip_path = self.jobs_dir / f"{job_id}.result.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for p in changed:
                    z.write(p, p.relative_to(job_dir).as_posix())
            self.results[job_id] = zip_path
            sent = True
        print(f"    Job {job_id} done (exit {code}); results {'sent' if sent else 'not sent'}. Waiting for jobs.",
              flush=True)
        try:
            self.wfile.write(f"\n__RELAY__ {job_id} {code} {'sent' if sent else 'withheld'} {len(changed)}\n".encode())
        except ConnectionError:
            pass


def worker(args):
    jobs = Path(args.jobs).resolve()
    jobs.mkdir(parents=True, exist_ok=True)
    token_file = jobs.parent / "relay-token.txt"
    token = os.environ.get("GPU_RELAY_TOKEN")
    if not token:
        if not token_file.exists():
            token_file.write_text(secrets.token_urlsafe(16))
        token = token_file.read_text().strip()
    Worker.name, Worker.token, Worker.gpu = args.name, token, gpu_name()
    Worker.jobs_dir, Worker.approve_timeout = jobs, args.approve_timeout
    threading.Thread(target=read_console, daemon=True).start()
    print(f"GPU relay worker '{args.name}' on port {args.port}\n  GPU:   {Worker.gpu}\n  token: {token}"
          f"  (give this to the sender once)\n  jobs:  {jobs}\nWaiting for jobs. Ctrl+C to stop.", flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Worker).serve_forever()
    except KeyboardInterrupt:
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


def status(friend):
    try:
        conn, r = call(friend, "GET", "/status", timeout=4)
        data = json.loads(r.read())
        conn.close()
        if r.status == 401:
            return "wrong token", data
        return ("busy" if data["busy"] else "online"), data
    except (OSError, ValueError, http.client.HTTPException):
        return "offline", {}


def check_all(names, friends):
    with ThreadPoolExecutor(max(1, len(names))) as pool:
        return dict(zip(names, pool.map(lambda n: status(friends[n]), names)))


def cmd_friends(args):
    friends = load_friends()
    if not friends:
        sys.exit("No friends yet. Add one: gpu-relay add <name> <address> <token>")
    for name, (state, data) in check_all(list(friends), friends).items():
        detail = f"{data.get('os', '')}, {data['gpu']}" if "gpu" in data else friends[name]["host"]
        print(f"  {name:<14} {state:<11} {detail}")


def cmd_add(args):
    friends = load_friends()
    friends[args.name] = {"host": args.host, "port": args.port, "token": args.token}
    save_friends(friends)
    state, data = status(friends[args.name])
    print(f"Saved {args.name}. Right now: {state}" + (f" ({data['gpu']})" if "gpu" in data else ""))


def cmd_remove(args):
    friends = load_friends()
    if friends.pop(args.name, None) is None:
        sys.exit(f"No friend called {args.name}")
    save_friends(friends)
    print(f"Removed {args.name}.")


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


def cmd_send(args):
    friends = load_friends()
    names = args.to or list(friends)
    unknown = [n for n in names if n not in friends]
    if unknown:
        sys.exit(f"Unknown friend(s): {', '.join(unknown)}. See: gpu-relay friends")
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
        with zipfile.ZipFile(upload, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for p in project.rglob("*"):
                rel = p.relative_to(project)
                if p.is_file() and not SKIP.intersection(rel.parts):
                    z.write(p, rel.as_posix())
        size = upload.stat().st_size

        print(f"Asking {', '.join(free)} to run \"{args.command}\" on {project.name} ({mb(size)})...", flush=True)
        winner = broadcast(free, friends, {"sender": f"{getpass.getuser()}@{socket.gethostname()}",
                                           "project": project.name, "size": size, "command": args.command})
        if not winner:
            sys.exit("Nobody accepted the job.")
        name, ticket = winner
        friend = friends[name]
        print(f"{name} accepted. Uploading and running on {states[name][1].get('gpu', 'their GPU')}...\n", flush=True)

        status_line = None
        with open(upload, "rb") as f:
            conn, r = call(friend, "POST", "/run", body=f, headers={
                "X-Ticket": ticket, "Content-Length": str(size), "Content-Type": "application/zip"})
            if r.status != 200:
                sys.exit(f"{name}'s laptop refused the upload: {r.read().decode(errors='replace')}")
            for line in r:
                if line.startswith(b"__RELAY__ "):
                    status_line = line.decode().split()[1:]
                else:
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()
            conn.close()

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


def main():
    ap = argparse.ArgumentParser(prog="gpu-relay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"gpu-relay {__version__}")
    sub = ap.add_subparsers(dest="mode", required=True)

    w = sub.add_parser("worker", help="share this laptop's GPU")
    w.add_argument("--name", default=socket.gethostname(), help="name shown to senders")
    w.add_argument("--host", default="0.0.0.0")
    w.add_argument("--port", type=int, default=PORT)
    w.add_argument("--jobs", default=str(HOME / "jobs"), help="where jobs run (default ~/.gpu-relay/jobs)")
    w.add_argument("--approve-timeout", type=int, default=180, help="seconds before an unanswered request counts as no")
    w.set_defaults(fn=worker)

    a = sub.add_parser("add", help="save a friend's laptop")
    a.add_argument("name")
    a.add_argument("host", help="Tailscale name or 100.x.y.z address")
    a.add_argument("token")
    a.add_argument("--port", type=int, default=PORT)
    a.set_defaults(fn=cmd_add)

    r = sub.add_parser("remove", help="forget a friend")
    r.add_argument("name")
    r.set_defaults(fn=cmd_remove)

    sub.add_parser("friends", help="show who is online").set_defaults(fn=cmd_friends)

    s = sub.add_parser("send", help="run a project on whoever accepts first")
    s.add_argument("project", help="project folder to send")
    s.add_argument("command", help='command to run inside it, e.g. "python train.py"')
    s.add_argument("--to", nargs="+", metavar="NAME", help="only ask these friends")
    s.set_defaults(fn=cmd_send)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
