#!/usr/bin/env python3
"""Run the official TRAM/EMDB evaluation without executing on import."""

from pathlib import Path
import runpy
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--input_dir', default='results/emdb')
    parser.add_argument('--data-root', default='data/emdb')
    parser.parse_known_args()
    implementation = Path(__file__).with_name('_run_eval_impl.py')
    runpy.run_path(str(implementation), run_name='__main__')


if __name__ == '__main__':
    main()
