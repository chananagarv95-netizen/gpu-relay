# gpu-relay

**Run your project on a friend's GPU laptop.** Keep a list of friends. When you send a job, every friend who is online and free gets a request on their screen. **The first to press Enter runs it**, and the request disappears for everyone else. Training output streams to you live. When it finishes, that friend presses Enter again and the result files land in your project folder.

- **Platforms:** Windows, macOS and Linux, for both sending and sharing a GPU.
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

**C. Install gpu-relay:**
```bat
py -m pip install https://github.com/chananagarv95-netizen/gpu-relay/archive/refs/heads/main.zip
```

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

**C. Install gpu-relay:**
```bash
pip install https://github.com/chananagarv95-netizen/gpu-relay/archive/refs/heads/main.zip
```

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
pip install https://github.com/chananagarv95-netizen/gpu-relay/archive/refs/heads/main.zip
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
py -m pip install --upgrade --force-reinstall https://github.com/chananagarv95-netizen/gpu-relay/archive/refs/heads/main.zip
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
pip install --upgrade --force-reinstall https://github.com/chananagarv95-netizen/gpu-relay/archive/refs/heads/main.zip
brew update && brew upgrade python@3.12 && brew upgrade --cask tailscale-app
```
GPU friends also run:
```bash
pip install --upgrade torch torchvision torchaudio numpy pandas scikit-learn matplotlib tqdm
```

Check your version with `gpu-relay --version`.

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
- **Without installing:** download `gpu_relay.py` and run `python gpu_relay.py <same arguments>`.
- **Tests:** `python3 test_relay.py` runs two workers and a sender on one machine.

## License
MIT
