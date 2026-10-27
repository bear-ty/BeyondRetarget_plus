#!/usr/bin/env python3
"""Estimate EMDB SMPL trajectories using the official TRAM/VIMO model."""

from pathlib import Path
import runpy
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--output_dir', default='results/emdb/smpl')
    parser.add_argument('--efficient', action='store_true')
    parser.add_argument('--data-root', default='data/emdb')
    parser.parse_known_args()
    implementation = Path(__file__).with_name('_run_smpl_impl.py')
    runpy.run_path(str(implementation), run_name='__main__')


if __name__ == '__main__':
    main()
