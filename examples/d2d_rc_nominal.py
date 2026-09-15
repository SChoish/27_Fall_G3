"""Evaluate the fixed nominal design with the RC channel; no optimization."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2d_sizing.circuit import Evaluator, NOMINAL, default_suite, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channel-model', choices=('rc', 'rlc'), default='rc')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve prior measurements')
    evaluator = Evaluator(default_suite(channel_model=args.channel_model), args.output)
    result = evaluator.evaluate(NOMINAL)
    report = {'channel_model': args.channel_model, 'scope': 'Fixed nominal, development only',
              'design': NOMINAL.tolist(), 'result': result, 'cost': evaluator.accounting()}
    write_json(args.output/'summary.json', report)
    print(json.dumps(report, indent=2))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
