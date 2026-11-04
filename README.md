# gpu-relay

**Run your project on a friend's GPU laptop, or on a whole computer lab.** Keep a list of friends. When you send a job, every friend who is online and free gets a request on their screen. **The first to press Enter runs it**, and the request disappears for everyone else. Training output streams to you live. When it finishes, that friend presses Enter again and the result files land in your project folder.

- **Platforms:** Windows, macOS and Linux, for both sending and sharing a GPU.
- **Lab mode:** on PCs you have permission to use, jobs run automatically in the background, many at once, with a live dashboard. See [section 5](#5-lab-pcs-automatic-mode).
- **Size:** one Python file, no dependencies beyond Python itself.
- **Safety:**
  - Nothing runs without the owner pressing Enter.
  - Unanswered requests count as "no" after 3 minutes.
  - Only people with your token can send you requests.

```
> gpu-relay friends
  ravi           online      Windows, NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB
  aman           busy        Windows, NVIDIA GeForce RTX 4060 Laptop GPU, 8192 MiB
  kunal          offline     kunal-laptop

> gpu-relay send ./my-project "python train.py --epochs 10"
  aman: busy
  kunal: offline
Asking ravi to run "python train.py --epochs 10" on my-project (12.4 MB)...
ravi accepted. Uploading and running on NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB...
epoch 1 loss 0.91
...
Job 20261101-231500-ab12 on ravi exited with code 0. 3 result file(s) saved to my-project/relay-results/20261101-231500-ab12
```

---

## 1. Install

Pick the section for your computer. **Senders** only need steps A, B and C. **Friends sharing a GPU** also need step D.

### Windows (Command Prompt)

**A. Python 3.12 and Tailscale** (Tailscale is the free private network that connects the laptops across any Wi-Fi):
```bat
winget install -e --id Python.Python.3.12
winget install -e --id Tailscale.Tailscale
```
Close and reopen Command Prompt afterwards. Open Tailscale from the Start menu and sign in.

**B. Update pip:**
```bat
py -m pip install --upgrade pip
```

**C. Install gpu-relay.** Put the file `gpu_relay-0.2.0-py3-none-any.whl` you were sent in your Downloads folder, then:
```bat
py -m pip install "%USERPROFILE%\Downloads\gpu_relay-0.2.0-py3-none-any.whl"
```
Check it with `gpu-relay --version`.

**D. Only if you share an NVIDIA GPU:**
1. Update the graphics driver with the **NVIDIA App** (nvidia.com/en-us/software/nvidia-app) or from nvidia.com/drivers, then restart.
2. Check the GPU is visible:
   ```bat
   nvidia-smi
   ```
3. Install PyTorch with CUDA:
   ```bat
   py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
4. Check PyTorch can use the GPU. It should print `True` and the GPU name:
   ```bat
   py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
5. Install what your friends' projects use, for example:
   ```bat
   py -m pip install numpy pandas scikit-learn matplotlib tqdm
   ```

If `cu126` fails, get the current command from pytorch.org/get-started/locally (Windows → Pip → CUDA).

### macOS (Terminal)

**A. Homebrew, Python 3.12 and Tailscale.** Skip the Homebrew line if `brew` already works.
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
brew install --cask tailscale-app
```
Open Tailscale from Applications and sign in.

**B. A private Python environment for gpu-relay.** macOS blocks installing into Homebrew's Python directly.
```bash
python3.12 -m venv ~/.gpu-relay/venv
source ~/.gpu-relay/venv/bin/activate
pip install --upgrade pip
echo 'source ~/.gpu-relay/venv/bin/activate' >> ~/.zshrc
```
The last line makes every new Terminal window use this environment.

**C. Install gpu-relay.** Put the file `gpu_relay-0.2.0-py3-none-any.whl` you were sent in your Downloads folder, then:
```bash
pip install ~/Downloads/gpu_relay-0.2.0-py3-none-any.whl
```
Check it with `gpu-relay --version`.

**D. Only if you share your Mac's GPU** (Apple Silicon, M1–M4). PyTorch uses it as device `mps`.
```bash
pip install torch torchvision torchaudio
python -c "import torch; print(torch.backends.mps.is_available())"
pip install numpy pandas scikit-learn matplotlib tqdm
```
The check should print `True`.

### Linux
```bash
python3 -m venv ~/.gpu-relay/venv && source ~/.gpu-relay/venv/bin/activate
pip install ~/Downloads/gpu_relay-0.2.0-py3-none-any.whl
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```
For NVIDIA GPUs, install PyTorch using the selector on pytorch.org.

---

## 2. Connect with Tailscale (once per friend)

1. Everyone installs Tailscale and signs in (step A above).
2. Each friend shares their laptop with you: open **login.tailscale.com/admin/machines** → click their laptop → **Share** → enter your email → you accept the invite.
3. Find their laptop's address on the same Machines page. It's a name like `ravi-laptop` or an IP like `100.101.102.103`.

---

## 3. Use it

### Friend sharing a GPU
```
gpu-relay worker --name ravi
```
- It prints a **token**. Send it to the sender once; it stays the same (saved in `~/.gpu-relay/relay-token.txt`).
- On Windows, if the firewall asks, allow **Private networks**.
- Leave the window open while you're happy to share. Each request shows who sent it and the exact command:
  ```
  === Job request from garv@Garvs-MacBook-Air (100.64.0.7)
      project: my-project (12.4 MB)
      command: python train.py --epochs 10
  >>> Run this job on your GPU?
      Enter = yes, n + Enter = no (no answer in 180s = no)
  ```
- Press **Ctrl+C** to stop sharing.

### Sender
```
gpu-relay add ravi <ravi's Tailscale address> <ravi's token>     # once per friend
gpu-relay friends                                                 # who is online / busy / offline
gpu-relay send ./my-project "python train.py"                     # ask everyone free; first yes wins
gpu-relay send ./my-project "python train.py" --to ravi aman      # ask only these friends
gpu-relay remove aman
```

**If `gpu-relay` is "not recognized" on Windows,** use `py -m gpu_relay` instead, for example `py -m gpu_relay worker --name ravi`.

### Make your code work on any friend's GPU
```python
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
```
Save outputs such as checkpoints, plots and logs into the project folder. Every file the job creates or changes comes back to you.

---

## 4. Update everything

### Windows
```bat
py -m pip install --upgrade pip
py -m pip install --upgrade --force-reinstall "%USERPROFILE%\Downloads\gpu_relay-<new version>-py3-none-any.whl"
winget upgrade -e --id Tailscale.Tailscale
winget upgrade -e --id Python.Python.3.12
```
GPU friends also run:
```bat
py -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
py -m pip install --upgrade numpy pandas scikit-learn matplotlib tqdm
```
Update the NVIDIA driver from the NVIDIA App.

### macOS
```bash
source ~/.gpu-relay/venv/bin/activate
pip install --upgrade pip
pip install --upgrade --force-reinstall ~/Downloads/gpu_relay-<new version>-py3-none-any.whl
brew update && brew upgrade python@3.12 && brew upgrade --cask tailscale-app
```
GPU friends also run:
```bash
pip install --upgrade torch torchvision torchaudio numpy pandas scikit-learn matplotlib tqdm
```

For gpu-relay itself, use the new `.whl` file you were sent. Check your version with `gpu-relay --version`.

---

## 5. Lab PCs (automatic mode)

For computers you have **permission** to use, such as college lab PCs. Nobody has to press Enter. Each PC runs one job at a time in the background, and you follow everything from a live dashboard on your Mac.

**What a lab PC does, and what it never does** (useful for the permission letter):
- **Account:** jobs run as Windows' low-privilege **LOCAL SERVICE** account. That isn't an administrator, so it can't install software or change settings. The admin password is only needed once, to install.
- **Priority:** jobs run at **below-normal priority**, so a student using the PC comes first.
- **Folder:** everything lives in `C:\gpu-relay`. Students can't open that folder, so they can't read the token.
- **Who can send jobs:** only someone with your token, and only over your private Tailscale network. The firewall rule accepts Tailscale addresses (`100.64.0.0/10`) only.
- **Time limit:** every job has one (default 90 minutes), and it is stopped when that runs out.
- **Sleep:** the PC is kept awake while a job runs, and normal sleep comes back after.
- **Pause or remove anytime:** see the end of this section.

### One-time preparation (on your Mac)

1. **Make a lab token** (one password shared by all lab PCs). Keep it secret and never put it on GitHub:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(24))"
   ```
2. **Make a Tailscale auth key** so the PCs can join your network without anyone signing in.
   - Go to login.tailscale.com/admin/settings/keys → **Generate auth key**.
   - Turn **Reusable** on and leave tags empty. Untagged devices are free and unlimited, while the free plan allows only 50 tagged ones.
3. **Copy to a pendrive:** `dist/gpu_relay-0.2.0-py3-none-any.whl`, `lab/install-lab-pc.ps1` and `lab/uninstall-lab-pc.ps1`, all in one folder.

### On each lab PC (once, by whoever has the admin password)

1. Plug in the pendrive.
2. Open **PowerShell as administrator** (Start → type PowerShell → right-click → Run as administrator).
3. Run:
   ```powershell
   cd E:\          # the pendrive's drive letter
   powershell -ExecutionPolicy Bypass -File .\install-lab-pc.ps1 -Token <lab token> -AuthKey <tskey-auth-...>
   ```

It takes about 5–10 minutes, mostly downloading PyTorch. It finishes with `OK: {"name": "lab1-pc07", ...}`. The PC's name is its Windows computer name in lower case. The worker then starts by itself at every boot, with no window.

- **Other libraries:** add `-Packages "numpy pandas opencv-python"` to install the libraries your jobs need. The default is numpy, pandas, scikit-learn, matplotlib and tqdm.
- **"Reset on restart" software** (Deep Freeze, Reboot Restore and similar) wipes everything at each reboot. The lab admin must **thaw** the PC (or exclude `C:\gpu-relay` and the Python, Tailscale and scheduled-task changes) before installing, then freeze it again.
- **Tailscale re-login:** Tailscale signs devices out after 180 days. To avoid that, on login.tailscale.com/admin/machines open each lab PC's **⋯** menu → **Disable key expiry**.
- **Same network instead of Tailscale:** if you'll always be on the lab's network, install with `-AllowLan`. Tailscale is still needed to reach the PCs from anywhere else.

### Add the PCs (on your Mac)

```bash
gpu-relay add-many <lab token> lab1-pc{01..30}       # zsh expands this to lab1-pc01 … lab1-pc30
gpu-relay friends --to "lab1-*"                       # which ones are online right now
gpu-relay dashboard --to "lab*"                       # live page: which PCs are online
```

### Run jobs

Put one command per line in a text file. Each line becomes one job on one free PC:
```text
# jobs.txt
python train.py --lr 0.1
python train.py --lr 0.01
python train.py --lr 0.001
python evaluate.py --fold 1
```
```bash
gpu-relay batch ./my-project jobs.txt --to "lab1-*" --time-limit 90
```

**The dashboard** opens at http://localhost:8766 and shows:
- every PC: online and free, running job N, or offline with when it was last seen
- every job: status, time used against the time limit, progress, and latest output line
- full output: click a job

Progress is the last `NN%` your program printed. tqdm bars work too.

**When jobs come back:** each job's files arrive as soon as it finishes, in `my-project/relay-results/batch-<date>/job001-lab1-pc07/`. Each job's full output is saved in `job001.log`.

**What happens when things go wrong:**

| Situation | What gpu-relay does |
|---|---|
| A student shuts the PC down mid-job | After 5 minutes unreachable (or straight away if the PC restarts), the job runs again on another free PC. It gives up after 3 PCs. |
| A job reaches `--time-limit` | It is stopped, and the files it saved so far (checkpoints!) come back. Status: `time limit`. |
| A job crashes | Status: `failed`, with its output in the log. It isn't re-run, since it would only crash again. |
| Your Mac sleeps or loses Wi-Fi | Lab jobs keep running. When the Mac is back, the batch carries on. |
| You close the Terminal window | Jobs keep running. Run `gpu-relay batch --resume my-project/relay-results/batch-<date>` to collect them. |
| You press Ctrl+C | The running jobs are cancelled on the lab PCs. Use `--resume` later to run the unfinished ones again. |

**Fit jobs into a lab session.** Two lectures are about 100 minutes, so the default `--time-limit` is 90. Keep each job well under that, for example a few epochs or one setting per line. Also save a checkpoint into the project folder every epoch, so even a stopped job sends something useful back.

**CPU-only PCs.** Lab PCs usually have no NVIDIA GPU, so PyTorch runs on the CPU, and the device code in section 3 picks `cpu` automatically. One lab i5 is roughly as fast as a MacBook Air. The gain comes from running 30 jobs at once.

### Pause or remove (on the lab PC, as administrator)

```powershell
Disable-ScheduledTask -TaskName gpu-relay; Stop-ScheduledTask -TaskName gpu-relay    # pause
Enable-ScheduledTask -TaskName gpu-relay; Start-ScheduledTask -TaskName gpu-relay    # resume
powershell -ExecutionPolicy Bypass -File .\uninstall-lab-pc.ps1                       # remove everything
powershell -ExecutionPolicy Bypass -File .\uninstall-lab-pc.ps1 -RemoveTailscale      # ...and Tailscale
```
The worker's own log is `C:\gpu-relay\worker.log`.

---

## Notes
- **What's uploaded:** the project folder, except `.git`, `node_modules`, `__pycache__`, `.venv`/`venv` and `relay-results`. It goes only to the friend who accepted.
- **What comes back:** only files the job created or changed, saved in `relay-results/<job-id>/`.
- **The command:**
  - `python …` runs with the Python that gpu-relay is installed in, so it uses that laptop's PyTorch.
  - Commands are a single program; `&&` and pipes are not supported, so for several steps write a script.
- **No answer:** after 180 s a request counts as "no". The friend can change this with `--approve-timeout`.
- **Stopping:** Ctrl+C on the sender stops the job on the friend's laptop.
- **Job files** stay on the friend's laptop in `~/.gpu-relay/jobs/<job-id>/`. They can delete old ones.
- **GPU memory:** a laptop RTX 3050 has 4–6 GB. Use small batch sizes and mixed precision (`torch.autocast`).
- **Without installing:** with just the file `gpu_relay.py`, run `python gpu_relay.py <same arguments>`.
- **Building the install file (maintainer):** `python3 -m pip wheel . --no-deps -w dist` creates `dist/gpu_relay-<version>-py3-none-any.whl`. Bump `version` in `pyproject.toml` and `__version__` in `gpu_relay.py` first.
- **Tests:** `python3 test_relay.py` (friends, Enter-to-approve) and `python3 test_lab.py` (lab mode: batch, time limit, shut-down PC, resume, Ctrl+C) run workers and a sender on one machine.

## License
MIT
