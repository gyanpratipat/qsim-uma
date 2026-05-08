"""
thermal_monitor.py

Verify that thermal monitoring works before running a thermally isolated
benchmark.  Also usable as a live pressure watch during runs.

macOS 15+ exposes thermal pressure (Nominal / Moderate / Heavy / Critical)
rather than a raw die temperature.  "Nominal" means the system is cool
enough to proceed.

Usage:
  python3 thermal_monitor.py            # single reading + sudo check
  python3 thermal_monitor.py --watch    # poll every 10s until Ctrl-C
"""

import subprocess, sys, time, argparse


def read_thermal_pressure():
    """Returns pressure string ('Nominal', 'Moderate', 'Heavy', 'Critical')
    or None if powermetrics / sudo is not configured."""
    try:
        r = subprocess.run(
            ['sudo', 'powermetrics', '--samplers', 'thermal', '-i', '500', '-n', '1'],
            capture_output=True, text=True, timeout=20)
        for line in r.stdout.splitlines():
            if 'current pressure level' in line.lower():
                return line.split(':')[1].strip()
    except Exception:
        pass
    return None


def check():
    pressure = read_thermal_pressure()
    if pressure is None:
        print("\n" + "!"*60)
        print("  FAIL: Cannot read thermal pressure via powermetrics.")
        print("  powermetrics requires passwordless sudo access.")
        print()
        print("  Fix — run this command:")
        print('    echo "ALL ALL=(root) NOPASSWD: /usr/bin/powermetrics" \\')
        print("      | sudo tee /etc/sudoers.d/powermetrics")
        print("!"*60 + "\n")
        return None
    status = "✓ ready to benchmark" if pressure == 'Nominal' else "⚠ still warm — wait before benchmarking"
    print(f"  Thermal pressure: {pressure}  {status}")
    return pressure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch',    action='store_true',
                        help='Poll every --interval seconds (Ctrl-C to stop)')
    parser.add_argument('--interval', type=int, default=10,
                        help='Poll interval in seconds for --watch (default: 10)')
    args = parser.parse_args()

    if not args.watch:
        pressure = check()
        if pressure is None:
            sys.exit(1)
        return

    print(f"Watching thermal pressure (interval {args.interval}s) …")
    print("Press Ctrl-C to stop.\n")
    try:
        while True:
            pressure = read_thermal_pressure()
            if pressure is None:
                print("  [ERROR] Cannot read pressure — check sudo config")
            else:
                marker = "✓" if pressure == 'Nominal' else "⚠"
                print(f"  {time.strftime('%H:%M:%S')}  {pressure}  {marker}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == '__main__':
    main()
