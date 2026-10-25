"""A live status line for slow steps, so a quiet terminal still shows the work moving.

Standard library only: install.py uses it before the packages are installed.
"""
import os
import shutil
import sys
import threading
import time

SPIN = "|/-\\"
PARTIAL = (".incomplete", ".part")  # names Hugging Face and snapshot downloads give files still arriving


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def dir_size(path) -> int:
    """Total size of the files under path (0 if it does not exist), counting files still being written.

    Windows' folder listing can show a size from before the file was last opened (0 for a new one), so
    files listed as empty, and partial downloads, are asked directly; asking every file would take
    seconds in site-packages.
    """
    total, stack = 0, [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            size = e.stat(follow_symlinks=False).st_size
                            if not size or e.name.endswith(PARTIAL):
                                size = os.stat(e.path).st_size
                            total += size
                    except OSError:
                        pass  # a file removed while we looked
        except OSError:
            pass
    return total


class Status:
    """`with Status("downloading"):` redraws "  | 1m 05s  downloading" four times a second.

    The text can be a function, called on every redraw, for counters that change on their own.
    When output goes to a file instead of a terminal, a "still working" line is printed every 30 s.
    """

    def __init__(self, text, quiet_every: float = 30):
        self.text, self.quiet_every = text, quiet_every
        self.tty = sys.stdout.isatty()
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._shown = 0

    def set(self, text) -> None:
        self.text = text

    def elapsed(self) -> float:
        return time.time() - self.t0

    def print(self, line: str) -> None:
        """Print a lasting line above the status line."""
        with self._lock:
            self._clear()
            print(line, flush=True)

    def _message(self) -> str:
        try:
            text = self.text() if callable(self.text) else self.text
        except Exception:  # a counter that failed this time; the clock still shows the work is alive
            text = ""
        return f"{fmt_time(self.elapsed()):>6}  {text}"

    def _clear(self) -> None:
        if self._shown:
            sys.stdout.write("\r" + " " * self._shown + "\r")
            self._shown = 0

    def _draw(self, frame: int) -> None:
        width = shutil.get_terminal_size((80, 24)).columns - 1
        line = f"  {SPIN[frame % len(SPIN)]} {self._message()}"[:width]
        with self._lock:
            sys.stdout.write("\r" + line + " " * max(0, self._shown - len(line)))
            sys.stdout.flush()
            self._shown = len(line)

    def _run(self) -> None:
        frame, last = 0, time.time()
        while not self._stop.wait(0.25 if self.tty else 1):
            if self.tty:
                self._draw(frame)
                frame += 1
            elif time.time() - last >= self.quiet_every:
                print(f"  still working: {self._message()}", flush=True)
                last = time.time()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._clear()
            sys.stdout.flush()
