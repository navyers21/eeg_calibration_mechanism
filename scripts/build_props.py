"""Compute Karan's six low-level image properties for every THINGS image once and cache them on Drive.
Needs the images (run download_data.py with --images in the same session).

    python scripts/build_props.py --config C
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eegalign import env, props  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
env.load(p.parse_args().config)
print(props.build_props_cache())
