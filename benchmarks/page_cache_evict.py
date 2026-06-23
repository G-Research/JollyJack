#!/usr/bin/env python3
"""Measure how long it takes to evict a file from the page cache.

Creates temporary files, populates the page cache by reading them, then
evicts them with posix_fadvise(POSIX_FADV_DONTNEED) and times the call.
Tracks global page-cache size via /proc/meminfo. A --sweep mode grows the
total page cache in steps and measures eviction time at each level, to test
whether eviction gets slower as the cache grows.

Linux only. WSL2 note: WSL2 does not faithfully account page cache in
/proc/meminfo and may not honor POSIX_FADV_DONTNEED, so numbers from WSL2 are
not representative. Run on the target (L4) nodes.
"""

import argparse
import os
import random
import sys
import threading
import time
from collections import deque

PAGE = os.sysconf("SC_PAGE_SIZE")


def parse_size(s):
    """Parse '1G', '512M', '64K', '4096' into bytes."""
    s = s.strip().upper()
    mult = 1
    if s and s[-1] in "KMGT":
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[s[-1]]
        s = s[:-1]
    return int(float(s) * mult)


def fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def meminfo():
    """Return selected /proc/meminfo fields in bytes."""
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) * 1024
    return out


def create_file(path, size, chunk=64 * 1024 * 1024, label=None, on_write=None):
    """Create a file of exactly `size` bytes.

    Reuse the file as-is if it already exists with exactly `size` bytes.
    If `label` is given, print write progress on a single updating line.
    If `on_write` is given, call it with each delta of bytes written (or with
    `size` on reuse) instead of printing; the caller owns the output line.
    Return True if the file was written, False if an existing same-size file
    was reused.
    """
    try:
        if os.path.getsize(path) == size:
            if on_write:
                on_write(size)
            elif label:
                print(f"  {label}: reuse {fmt_bytes(size)} {path}")
            return False
    except FileNotFoundError:
        pass
    written = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        while written < size:
            # Fresh random bytes per chunk so distinct blocks defeat any
            # filesystem compression or dedup that would otherwise keep the
            # file from really occupying the page cache.
            n = os.write(fd, random.randbytes(min(chunk, size - written)))
            written += n
            if on_write:
                on_write(n)
            elif label:
                pct = 100 * written / size if size else 100
                print(f"\r  {label}: {fmt_bytes(written)} / "
                      f"{fmt_bytes(size)} ({pct:.0f}%)", end="", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
        if label and not on_write:
            print()
    return True


class RateWindow:
    """Rolling byte throughput over a trailing time window (seconds)."""

    def __init__(self, window=10.0):
        self.window = window
        self.samples = deque()  # (timestamp, cumulative_bytes)

    def update(self, t, done):
        self.samples.append((t, done))
        cutoff = t - self.window
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def rate(self):
        if len(self.samples) < 2:
            return 0.0
        t0, b0 = self.samples[0]
        t1, b1 = self.samples[-1]
        dt = t1 - t0
        return (b1 - b0) / dt if dt > 0 else 0.0


def run_with_progress(ts, get_done, total, label, progress=True):
    """Start threads `ts`, print a single updating progress line, and join.

    `get_done()` returns the cumulative bytes processed so far. The live line
    shows percent and throughput averaged over the last 10 seconds; the final
    line shows the overall average. Returns the elapsed seconds.
    """
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    if progress:
        win = RateWindow(10.0)
        while any(t.is_alive() for t in ts):
            now = time.perf_counter()
            d = get_done()
            win.update(now, d)
            pct = 100 * d / total if total else 100
            print(f"\r  {label}: {fmt_bytes(d)} / {fmt_bytes(total)} "
                  f"({pct:.0f}%, {fmt_bytes(win.rate())}/s)",
                  end="", flush=True)
            time.sleep(0.1)
    for t in ts:
        t.join()
    secs = time.perf_counter() - t0
    if progress and total:
        rate = total / secs if secs > 0 else float("inf")
        print(f"\r  {label}: {fmt_bytes(total)} / {fmt_bytes(total)} "
              f"(100%) in {secs:.2f}s (avg {fmt_bytes(rate)}/s)")
    return secs


def read_sequential(fd, size, chunk=8 * 1024 * 1024):
    off = 0
    while off < size:
        data = os.pread(fd, min(chunk, size - off), off)
        if not data:
            break
        off += len(data)


def populate(paths, workers, progress=True, chunk=8 * 1024 * 1024):
    """Read the given files into the page cache, up to `workers` at a time.

    os.pread releases the GIL, so threads read concurrently. When `progress`
    is set, print a combined bytes-read line that updates as reads proceed.
    """
    sizes = {p: os.path.getsize(p) for p in paths}
    total = sum(sizes.values())
    done = 0
    lock = threading.Lock()
    sem = threading.Semaphore(max(1, workers))

    def work(p):
        nonlocal done
        with sem:
            fd = os.open(p, os.O_RDONLY)
            try:
                off = 0
                size = sizes[p]
                while off < size:
                    data = os.pread(fd, min(chunk, size - off), off)
                    if not data:
                        break
                    off += len(data)
                    with lock:
                        done += len(data)
            finally:
                os.close(fd)

    def get_done():
        with lock:
            return done

    ts = [threading.Thread(target=work, args=(p,)) for p in paths]
    run_with_progress(ts, get_done, total, "reading", progress)


def create_files(paths, size, workers, progress=True):
    """Create the given files in parallel, up to `workers` at a time.

    os.write releases the GIL, so threads write concurrently. Existing
    same-size files are reused. When `progress` is set, print a combined
    bytes-written line. Returns the list of paths actually written this run.
    """
    total = size * len(paths)
    done = 0
    created = []
    lock = threading.Lock()
    sem = threading.Semaphore(max(1, workers))

    def on_write(n):
        nonlocal done
        with lock:
            done += n

    def work(p):
        with sem:
            wrote = create_file(p, size, on_write=on_write)
        if wrote:
            with lock:
                created.append(p)

    def get_done():
        with lock:
            return done

    ts = [threading.Thread(target=work, args=(p,)) for p in paths]
    run_with_progress(ts, get_done, total, "writing", progress)
    return created


def evict(path):
    """Evict `path` via POSIX_FADV_DONTNEED. Return seconds in the call."""
    t0 = time.perf_counter()
    with open(path, "r") as f:
        fd = f.fileno()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        
    return time.perf_counter() - t0    


def measure_once(targets, workers=1):
    """Read the targets into cache, then evict them one at a time.

    Returns a per-file list of dicts. `cached_before` is the global page-cache
    size (/proc/meminfo Cached) captured just before that file is evicted, so
    it falls as earlier files are dropped.
    """
    populate(targets, workers)

    results = []
    for p in targets:
        size = os.path.getsize(p)
        cached_before = meminfo()["Cached"]
        secs = evict(p)
        results.append({
            "path": p,
            "size": size,
            "pages": (size + PAGE - 1) // PAGE,
            "cached_before": cached_before,
            "evict_secs": secs,
        })
    return results


def print_result(results, label=""):
    if label:
        print(f"[{label}]")
    for r in results:
        pages = r["pages"]
        secs = r["evict_secs"]
        thru = (r["size"] / secs) if secs > 0 else float("inf")
        ns_per_page = (secs * 1e9 / pages) if pages else float("nan")
        print(f"  {os.path.basename(r['path'])}: "
              f"{fmt_bytes(r['size'])} ({pages} pages)  "
              f"cache_before={fmt_bytes(r['cached_before'])}  "
              f"evict={secs * 1e3:.3f} ms "
              f"({ns_per_page:.0f} ns/page, {fmt_bytes(thru)}/s)")
    if len(results) > 1:
        tot_secs = sum(r["evict_secs"] for r in results)
        tot_size = sum(r["size"] for r in results)
        thru = (tot_size / tot_secs) if tot_secs > 0 else float("inf")
        print(f"  total: {fmt_bytes(tot_size)}  "
              f"evict={tot_secs * 1e3:.3f} ms ({fmt_bytes(thru)}/s)")
    m = meminfo()
    print(f"  meminfo Cached={fmt_bytes(m['Cached'])} "
          f"MemFree={fmt_bytes(m['MemFree'])}")


def cleanup(paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.environ.get("TMPDIR", "/tmp"),
                    help="directory for temp files (default: $TMPDIR or /tmp)")
    ap.add_argument("--file-size", default="1G", type=parse_size,
                    help="size of each target file (default 1G)")
    ap.add_argument("--num-files", default=1, type=int,
                    help="number of target files (default 1)")
    ap.add_argument("--workers", default=0, type=int,
                    help="parallel readers when populating the cache "
                         "(default 0 = one per target file)")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete any temp files at exit (by default "
                         "only files created this run are deleted; reused "
                         "files are always kept)")
    args = ap.parse_args()

    if sys.platform != "linux":
        print("This tool is Linux-only.", file=sys.stderr)
        return 2

    os.makedirs(args.dir, exist_ok=True)
    workers = args.workers if args.workers > 0 else args.num_files
    targets = [os.path.join(args.dir, f"pce_target_{i}.bin")
               for i in range(args.num_files)]
    created = []  # files written this run; reused files are left alone

    try:
        print(f"creating {args.num_files} target file(s) of "
              f"{fmt_bytes(args.file_size)} in {args.dir}")
        created += create_files(targets, args.file_size, workers)

        m0 = meminfo()
        print(f"start: Cached={fmt_bytes(m0['Cached'])} "
              f"MemTotal={fmt_bytes(m0['MemTotal'])} "
              f"MemFree={fmt_bytes(m0['MemFree'])}\n")

        r = measure_once(targets, workers)
        print_result(r)
    finally:
        if not args.keep:
            cleanup(created)
        elif created:
            print(f"\nkept {len(created)} temp file(s) in {args.dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
