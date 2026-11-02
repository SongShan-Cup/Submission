"""Short, offline queue/cancel/timeout demo. No GPU or platform API required."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=90)
    parser.add_argument('--exit-code', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.seconds <= 300 or not 0 <= args.exit_code <= 125:
        parser.error('seconds must be 0..300; exit-code must be 0..125')
    started = time.monotonic()
    print('TRIAL_START', json.dumps(vars(args)), flush=True)
    for second in range(args.seconds):
        time.sleep(1)
        if (second + 1) % 5 == 0:
            print('TRIAL_PROGRESS', second + 1, flush=True)
    result = dict(requested_seconds=args.seconds,
                  requested_exit_code=args.exit_code,
                  program_seconds=round(time.monotonic() - started, 3))
    Path('trial-result.json').write_text(json.dumps(result), encoding='utf-8')
    print('TRIAL_END', json.dumps(result), flush=True)
    return args.exit_code


if __name__ == '__main__':
    sys.exit(main())
