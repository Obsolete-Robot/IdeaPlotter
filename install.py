"""IdeaPlotter installer: python install.py (or install.bat / ./install.sh).

Sets up everything needed to run the map, and can be run again at any time
(finished steps are skipped):

  1. a Python environment in .venv/
  2. PyTorch, matched to your GPU (or the CPU), and the other packages
  3. the prebuilt game map (~440 MB from the project's GitHub release)
  4. how ideas get written: an OpenRouter or Anthropic API key, Claude Code, or later

Keys are saved in .env next to this file, which git ignores.
"""
import argparse
import collections
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ideaplotter.progress import Status, dir_size, fmt_bytes, fmt_time  # noqa: E402  (standard library only)

VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV_FILE = ROOT / ".env"
SITE_PACKAGES = VENV / "Lib/site-packages" if os.name == "nt" else None  # posix: found once the venv exists
PIP_TMP = VENV / "pip-tmp"  # pip's downloads land here, so their size can be shown as they arrive
TORCH_INDEX = "https://download.pytorch.org/whl/"
WRITER_KEYS = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "IDEAPLOTTER_BACKEND")


def step(n: int, text: str) -> None:
    print(f"\n[{n}/4] {text}")


def ask(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {hint} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "n", "no"):
            return answer.startswith("y")


def run(*args, **kw) -> int:
    return subprocess.call([str(a) for a in args], cwd=ROOT, **kw)


def make_venv() -> None:
    if VENV_PY.exists():
        print("  already set up")
        return
    with Status("creating .venv and installing pip into it (usually under a minute)"):
        made = subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, capture_output=True, text=True)
    if made.returncode != 0 or not VENV_PY.exists():
        print((made.stdout + made.stderr).strip())
        hint = (f"\n  On Ubuntu/Debian, install venv support first:  sudo apt install python{sys.version_info.major}."
                f"{sys.version_info.minor}-venv" if sys.platform.startswith("linux") else "")
        sys.exit(f"  Could not create the Python environment in {VENV}.{hint}")


def gpu() -> tuple[str, float] | None:
    """(name, CUDA compute capability) of the first NVIDIA GPU, if there is one."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        name, cap = [s.strip() for s in out[0].split(",")]
        return name, float(cap)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return ("NVIDIA GPU", 7.0)  # an older driver without compute_cap: the common case


def site_packages() -> Path:
    if SITE_PACKAGES:
        return SITE_PACKAGES
    return next(iter(sorted((VENV / "lib").glob("python*/site-packages"))), VENV / "lib")


SIZE = {"kB": 1e3, "MB": 1e6, "GB": 1e9}


def pip_install(*args) -> int:
    """pip install, with its output turned into a live status line (and shown in full if it fails)."""
    shutil.rmtree(PIP_TMP, ignore_errors=True)
    PIP_TMP.mkdir(parents=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TMP": str(PIP_TMP), "TEMP": str(PIP_TMP), "TMPDIR": str(PIP_TMP)}
    proc = subprocess.Popen([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "--progress-bar",
                             "off", *args], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    log = collections.deque(maxlen=80)
    now = {"what": "checking what is needed", "dir": None, "base": 0, "total": 0, "t0": time.time()}

    def measured() -> int:  # site-packages holds thousands of files: rescanned every 2 s, downloads twice a second
        if time.time() - now["scanned"] >= (0.5 if now["total"] else 2):
            got = max(0, dir_size(now["dir"]) - now["base"])
            if got != now["got"]:
                now["moved"] = time.time()
            now["got"], now["scanned"] = got, time.time()
        return now["got"]

    def text() -> str:
        if not now["dir"]:
            return now["what"]
        got = measured()
        if now["total"]:  # a download of known size
            got = min(got, now["total"])
            quiet = time.time() - now["moved"]
            speed = (f"no data for {fmt_time(quiet)}, still trying" if quiet >= 20
                     else f"{got / max(time.time() - now['t0'], 0.1) / 1e6:.1f} MB/s")
            return f"{now['what']}: {fmt_bytes(got)} of {fmt_bytes(now['total'])} ({got * 100 // now['total']}%, {speed})"
        return f"{now['what']}: {fmt_bytes(got)} written"

    def begin(what, folder=None, total=0):
        if now["total"] >= 20e6:  # say when each big download finished
            status.print(f"  downloaded {now['name']} ({fmt_bytes(now['total'])}) in "
                         f"{fmt_time(time.time() - now['t0'])}")
        now.update(what=what, dir=folder, base=dir_size(folder) if folder else 0, total=total, t0=time.time(),
                   got=0, scanned=time.time(), moved=time.time())

    with Status(text) as status:
        for line in proc.stdout:
            line = line.rstrip()
            log.append(line)
            if m := re.match(r"\s*Collecting (\S+)", line):
                begin(f"looking up {m[1]}")
            elif m := re.match(r"\s*Downloading (\S+)(?: \(([\d.]+) (kB|MB|GB)\))?", line):
                name = "-".join(urllib.parse.unquote(m[1].rsplit("/", 1)[-1]).split("-")[:2])
                total = int(float(m[2]) * SIZE[m[3]]) if m[2] else 0
                begin(f"downloading {name}", PIP_TMP, total)
                now["name"] = name
            elif m := re.match(r"\s*Using cached (\S+)", line):
                begin(f"using the copy of {m[1].split('-')[0]} downloaded earlier")
            elif m := re.match(r"\s*Building wheel for (\S+)", line):
                begin(f"building {m[1]}")
            elif m := re.match(r"Installing collected packages: (.*)", line):
                n = len(m[1].split(","))
                begin(f"unpacking {n} package{'s' * (n > 1)} into .venv", site_packages())
                now["installing"] = time.time()
            elif m := re.match(r"Successfully installed (.*)", line):
                begin("finishing")
                n = len(m[1].split())
                status.print(f"  installed {n} package{'s' * (n > 1)}"
                             + (f" in {fmt_time(time.time() - now['installing'])}" if "installing" in now else ""))
    code = proc.wait()
    shutil.rmtree(PIP_TMP, ignore_errors=True)
    if code != 0:
        print("\n".join(log))
    return code


def install_packages() -> None:
    with Status("checking for PyTorch"):
        have_torch = run(VENV_PY, "-c", "import torch", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if have_torch:
        print("  PyTorch already installed")
    else:
        found = None if sys.platform == "darwin" else gpu()
        if sys.platform == "darwin":
            print("  Mac: PyTorch runs on the CPU (searches for empty spots take a few minutes)")
            index = []
        elif found:
            # CUDA 12.6 builds cover GTX 900 through RTX 40 series; RTX 50 series needs CUDA 12.8
            cuda = "cu128" if found[1] >= 10 else "cu126"
            print(f"  {found[0]} found: installing PyTorch with CUDA ({cuda}), about 2.5 GB")
            index = ["--index-url", TORCH_INDEX + cuda]
        else:
            print("  No NVIDIA GPU found: installing PyTorch for the CPU (searches for empty spots take a few minutes)")
            index = ["--index-url", TORCH_INDEX + "cpu"]
        if pip_install("torch", *index) != 0:
            sys.exit("  Installing PyTorch failed (see the messages above). Run the installer again to retry.")
    print("  other packages")
    if pip_install("-r", "requirements.txt") != 0:
        sys.exit("  Installing packages failed (see the messages above). Run the installer again to retry.")


def download_map() -> None:
    if run(VENV_PY, "-m", "ideaplotter", "download") != 0:
        sys.exit("  The map did not download. Check your internet connection and run the installer again.")


def read_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                env[key.strip()] = value.strip()
    return env


def write_env(updates: dict) -> None:
    """Replace the writer settings in .env, keeping any other lines."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [l for l in lines if l.partition("=")[0].strip() not in WRITER_KEYS]
    lines += [f"{k}={v}" for k, v in updates.items()]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name != "nt":
        ENV_FILE.chmod(0o600)


def key_works(kind: str, key: str) -> bool | None:
    """True/False if the provider accepted/rejected the key, None if it could not be checked."""
    if kind == "openrouter":
        req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    else:
        req = urllib.request.Request("https://api.anthropic.com/v1/models",
                                     headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=20):
            return True
    except urllib.error.HTTPError as e:
        return False if e.code in (401, 403) else None
    except (urllib.error.URLError, OSError):
        return None


WRITERS = {
    "1": ("openrouter", "OpenRouter API key (recommended: about $0.03 per idea, https://openrouter.ai/keys)"),
    "2": ("anthropic", "Anthropic API key (https://console.anthropic.com)"),
    "3": ("claude", "Claude Code, using the Claude login on this computer (no key)"),
    "4": ("skip", "Decide later: browse the map now, set this up by running the installer again"),
}


def set_up_writer(choice: str | None) -> None:
    env = read_env()
    current = env.get("IDEAPLOTTER_BACKEND")
    if choice is None and current:
        label = {"openrouter": "an OpenRouter key", "api": "an Anthropic key", "cli": "Claude Code"}.get(current, current)
        if ask(f"  Ideas are written with {label}. Keep that?"):
            return
    if choice is None:
        print("  IdeaPlotter uses an AI model to write the game ideas. How should it reach one?")
        for n, (_, text) in WRITERS.items():
            print(f"    {n}. {text}")
        while (picked := input("  Choose 1-4: ").strip()) not in WRITERS:
            pass
        choice = WRITERS[picked][0]
    if choice == "skip":
        print("  Skipped: the map works, and writing ideas will ask you to set this up.")
        return
    if choice == "claude":
        write_env({"IDEAPLOTTER_BACKEND": "cli"})
        print("  Saved. Make sure Claude Code is installed and logged in (https://claude.com/claude-code).")
        return
    name = "OpenRouter" if choice == "openrouter" else "Anthropic"
    var = "OPENROUTER_API_KEY" if choice == "openrouter" else "ANTHROPIC_API_KEY"
    while True:
        key = getpass.getpass(f"  Paste your {name} API key (it stays hidden; Enter to cancel): ").strip()
        if not key:
            print("  Cancelled: nothing saved.")
            return
        works = key_works(choice, key)
        if works is False:
            print(f"  {name} did not accept that key. Try again.")
            continue
        if works is None:
            print(f"  Could not reach {name} to check the key; saving it anyway.")
        write_env({var: key, "IDEAPLOTTER_BACKEND": "openrouter" if choice == "openrouter" else "api"})
        print(f"  Saved to {ENV_FILE.name}" + (" (key checked)" if works else ""))
        return


def main() -> None:
    p = argparse.ArgumentParser(description="Set up IdeaPlotter.")
    p.add_argument("--writer", choices=["openrouter", "anthropic", "claude", "skip"],
                   help="how ideas are written, instead of asking")
    p.add_argument("--no-start", action="store_true", help="do not offer to start IdeaPlotter at the end")
    a = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with pip's output

    if sys.version_info < (3, 10):
        sys.exit(f"IdeaPlotter needs Python 3.10 or newer; this is {sys.version.split()[0]}. "
                 "Get a newer one at https://www.python.org/downloads/")
    print(f"Setting up IdeaPlotter in {ROOT}")
    if shutil.disk_usage(ROOT).free < 5e9:
        print("  Note: less than 5 GB free here; the full install needs about 4 GB.")

    step(1, "Python environment")
    make_venv()
    step(2, "Packages")
    install_packages()
    step(3, "The game map")
    download_map()
    step(4, "Writing ideas")
    set_up_writer(a.writer)

    start = "run.bat" if os.name == "nt" else "./run.sh"
    print(f"\nIdeaPlotter is ready. Start it any time with {start}.")
    if not a.no_start and ask("Start it now? The map opens in your browser."):
        print("Starting. The first start also downloads the embedding model (about 1.2 GB). Ctrl+C stops it.")
        sys.exit(run(VENV_PY, "-m", "ideaplotter", "serve", "--open"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
