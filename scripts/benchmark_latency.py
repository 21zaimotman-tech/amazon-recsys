"""Hit the live API and report average response time per component + total.
Required for the README ("average response time") and ANALYSIS.md latency table.

Usage:  python scripts/benchmark_latency.py http://localhost:8000 user1 user2 ...
        (or pipe a file of user_ids). Runs each user a few times, warms up first."""
import sys, statistics, requests


def benchmark(api, user_ids, repeats=5, n=10):
    comp = {}
    requests.get(f"{api}/health")                      # warm up
    for u in user_ids:
        for _ in range(repeats):
            r = requests.get(f"{api}/recommend/{u}?n={n}", timeout=15).json()
            for k, v in r.get("latency_ms", {}).items():
                if isinstance(v, (int, float)):
                    comp.setdefault(k, []).append(v)
    print(f"\nLatency over {len(user_ids)*repeats} calls (ms):")
    print(f"{'component':14s} {'mean':>8s} {'p95':>8s}")
    for k in sorted(comp):
        vals = sorted(comp[k]); p95 = vals[int(0.95 * (len(vals) - 1))]
        print(f"{k:14s} {statistics.mean(vals):8.2f} {p95:8.2f}")
    return comp


if __name__ == "__main__":
    api = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    users = sys.argv[2:] or [l.strip() for l in sys.stdin if l.strip()]
    benchmark(api, users)
