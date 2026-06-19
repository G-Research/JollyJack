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
import sys
import threading
import time

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


def create_file(path, size, chunk=64 * 1024 * 1024, label=None):
    """Create a file of exactly `size` bytes.

    If `label` is given, print write progress on a single updating line.
    Reuse the file as-is if it already exists with exactly `size` bytes.
    Return True if the file was written, False if an existing same-size file
    was reused.
    """
    try:
        if os.path.getsize(path) == size:
            if label:
                print(f"  {label}: reuse {fmt_bytes(size)} {path}")
            return False
    except FileNotFoundError:
        pass
    buf = b"\x5a" * min(chunk, size)
    written = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        while written < size:
            n = os.write(fd, buf[: min(chunk, size - written)])
            written += n
            if label:
                pct = 100 * written / size if size else 100
                print(f"\r  {label}: {fmt_bytes(written)} / "
                      f"{fmt_bytes(size)} ({pct:.0f}%)", end="", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
        if label:
            print()
    return True


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

    ts = [threading.Thread(target=work, args=(p,)) for p in paths]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    if progress:
        while any(t.is_alive() for t in ts):
            with lock:
                d = done
            pct = 100 * d / total if total else 100
            print(f"\r  reading: {fmt_bytes(d)} / {fmt_bytes(total)} "
                  f"({pct:.0f}%)", end="", flush=True)
            time.sleep(0.1)
    for t in ts:
        t.join()
    if progress and total:
        secs = time.perf_counter() - t0
        rate = total / secs if secs > 0 else float("inf")
        print(f"\r  reading: {fmt_bytes(total)} / {fmt_bytes(total)} "
              f"(100%) in {secs:.2f}s ({fmt_bytes(rate)}/s)")


def evict(path):
    """Evict `path` via POSIX_FADV_DONTNEED. Return seconds in the call."""
    fd = os.open(path, os.O_RDONLY)
    try:
        t0 = time.perf_counter()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return time.perf_counter() - t0
    finally:
        os.close(fd)


def make_filler(dir_path, total, prefix, chunk=1024 * 1024 * 1024):
    """Create filler files totalling `total` bytes and read them into cache.

    These stay resident (not evicted) so the total page cache grows by roughly
    `total`. Returns (all_paths, created_paths); created_paths are the ones
    written this run (existing same-size files are reused).
    """
    paths = []
    created = []
    made = 0
    i = 0
    while made < total:
        sz = min(chunk, total - made)
        p = os.path.join(dir_path, f"{prefix}_filler_{i}.bin")
        if create_file(p, sz,
                       label=f"filler {fmt_bytes(made)}/{fmt_bytes(total)}"):
            created.append(p)
        fd = os.open(p, os.O_RDONLY)
        try:
            read_sequential(fd, sz)
        finally:
            os.close(fd)
        paths.append(p)
        made += sz
        i += 1
    return paths, created


def measure_once(targets, workers=1):
    """Read the targets into cache, then evict them one at a time.

    Returns a per-file list of dicts. `cached_before` is the global page-cache
    size (/proc/meminfo Cached) captured just before that file is evicted, so
    it falls as earlier files are dropped.
    """
    # Start cold: writing the files leaves their pages cached, so drop them
    # first, otherwise the read finds everything already resident.
    for p in targets:
        evict(p)

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


def self_check():
    import tempfile
    d = tempfile.mkdtemp(prefix="pce_selfcheck_")
    p = os.path.join(d, "t.bin")
    create_file(p, 64 * 1024 * 1024)
    try:
        results = measure_once([p])
    finally:
        cleanup([p])
        os.rmdir(d)
    assert results and results[0]["evict_secs"] >= 0
    print_result(results, label="self-check")
    print("self-check OK")
    return 0


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
    ap.add_argument("--fill", default="0", type=parse_size,
                    help="pre-fill page cache with this much filler before "
                         "measuring (e.g. 50G)")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated fill sizes for a sweep, "
                         "e.g. 0,10G,50G,100G")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete any temp files at exit (by default "
                         "only files created this run are deleted; reused "
                         "files are always kept)")
    ap.add_argument("--self-check", action="store_true",
                    help="run a small correctness check and exit")
    args = ap.parse_args()

    if sys.platform != "linux":
        print("This tool is Linux-only.", file=sys.stderr)
        return 2

    if args.self_check:
        return self_check()

    os.makedirs(args.dir, exist_ok=True)
    workers = args.workers if args.workers > 0 else args.num_files
    targets = [os.path.join(args.dir, f"pce_target_{i}.bin")
               for i in range(args.num_files)]
    created = []  # files written this run; reused files are left alone

    try:
        print(f"creating {args.num_files} target file(s) of "
              f"{fmt_bytes(args.file_size)} in {args.dir}")
        for i, p in enumerate(targets):
            if create_file(p, args.file_size,
                           label=f"target {i + 1}/{len(targets)}"):
                created.append(p)

        m0 = meminfo()
        print(f"start: Cached={fmt_bytes(m0['Cached'])} "
              f"MemTotal={fmt_bytes(m0['MemTotal'])} "
              f"MemFree={fmt_bytes(m0['MemFree'])}\n")

        if args.sweep:
            levels = [parse_size(s) for s in args.sweep.split(",")]
            prev_fill = 0
            for lvl in levels:
                add = lvl - prev_fill
                if add > 0:
                    print(f"filling page cache to {fmt_bytes(lvl)} "
                          f"(+{fmt_bytes(add)}) ...")
                    _, c = make_filler(args.dir, add, f"pce_{lvl}")
                    created += c
                prev_fill = max(prev_fill, lvl)
                r = measure_once(targets, workers)
                print_result(r, label=f"fill={fmt_bytes(lvl)}")
                print()
        else:
            if args.fill > 0:
                print(f"filling page cache with {fmt_bytes(args.fill)} ...")
                _, c = make_filler(args.dir, args.fill, "pce")
                created += c
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
