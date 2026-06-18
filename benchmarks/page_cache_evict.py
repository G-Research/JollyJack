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


def create_file(path, size, chunk=64 * 1024 * 1024):
    """Create a file of exactly `size` bytes."""
    buf = b"\x5a" * min(chunk, size)
    written = 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        while written < size:
            n = os.write(fd, buf[: min(chunk, size - written)])
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def read_sequential(fd, size, chunk=8 * 1024 * 1024):
    off = 0
    while off < size:
        data = os.pread(fd, min(chunk, size - off), off)
        if not data:
            break
        off += len(data)


def populate(path):
    """Read the whole file so its pages enter the page cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        read_sequential(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


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

    Returns the list of created paths. These stay resident (not evicted) so
    the total page cache grows by roughly `total`.
    """
    paths = []
    made = 0
    i = 0
    while made < total:
        sz = min(chunk, total - made)
        p = os.path.join(dir_path, f"{prefix}_filler_{i}.bin")
        create_file(p, sz)
        fd = os.open(p, os.O_RDONLY)
        try:
            read_sequential(fd, sz)
        finally:
            os.close(fd)
        paths.append(p)
        made += sz
        i += 1
    return paths


def measure_once(targets):
    """Read the targets into cache, then evict them. Return a result dict."""
    total_size = sum(os.path.getsize(p) for p in targets)

    # Start cold: writing the files leaves their pages cached, so drop them
    # first, otherwise the read finds everything already resident.
    for p in targets:
        evict(p)

    for p in targets:
        populate(p)

    evict_secs = 0.0
    for p in targets:
        evict_secs += evict(p)

    total_pages = (total_size + PAGE - 1) // PAGE
    return {
        "total_size": total_size,
        "total_pages": total_pages,
        "evict_secs": evict_secs,
    }


def print_result(r, label=""):
    pages = r["total_pages"]
    secs = r["evict_secs"]
    thru = (r["total_size"] / secs) if secs > 0 else float("inf")
    ns_per_page = (secs * 1e9 / pages) if pages else float("nan")
    m = meminfo()
    head = f"[{label}] " if label else ""
    print(f"{head}target {fmt_bytes(r['total_size'])} ({pages} pages)")
    print(f"  evict time : {secs * 1e3:.3f} ms "
          f"({ns_per_page:.0f} ns/page, {fmt_bytes(thru)}/s)")
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
        r = measure_once([p])
    finally:
        cleanup([p])
        os.rmdir(d)
    assert r["evict_secs"] >= 0
    print_result(r, label="self-check")
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
    ap.add_argument("--fill", default="0", type=parse_size,
                    help="pre-fill page cache with this much filler before "
                         "measuring (e.g. 50G)")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated fill sizes for a sweep, "
                         "e.g. 0,10G,50G,100G")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete temp files at exit")
    ap.add_argument("--self-check", action="store_true",
                    help="run a small correctness check and exit")
    args = ap.parse_args()

    if sys.platform != "linux":
        print("This tool is Linux-only.", file=sys.stderr)
        return 2

    if args.self_check:
        return self_check()

    os.makedirs(args.dir, exist_ok=True)
    pid = os.getpid()
    targets = [os.path.join(args.dir, f"pce_{pid}_target_{i}.bin")
               for i in range(args.num_files)]
    filler = []

    try:
        print(f"creating {args.num_files} target file(s) of "
              f"{fmt_bytes(args.file_size)} in {args.dir}")
        for p in targets:
            create_file(p, args.file_size)

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
                    filler += make_filler(args.dir, add, f"pce_{pid}_{lvl}")
                prev_fill = max(prev_fill, lvl)
                r = measure_once(targets)
                print_result(r, label=f"fill={fmt_bytes(lvl)}")
                print()
        else:
            if args.fill > 0:
                print(f"filling page cache with {fmt_bytes(args.fill)} ...")
                filler += make_filler(args.dir, args.fill, f"pce_{pid}")
            r = measure_once(targets)
            print_result(r)
    finally:
        if not args.keep:
            cleanup(targets + filler)
        elif targets or filler:
            print(f"\nkept {len(targets + filler)} temp file(s) in {args.dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
