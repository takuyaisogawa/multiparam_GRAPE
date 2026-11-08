"""Portable data access; reruns never overwrite the released data."""
from pathlib import Path
from functools import lru_cache
import json, os
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'revision/data'
RESULTS=ROOT/'revision/results'
BUILD=Path(os.environ.get('NJP_OUTPUT_DIR',ROOT/'build')).resolve()

def output(name):
    path=BUILD/name
    path.mkdir(parents=True,exist_ok=True)
    return path

def load(name):
    return json.loads((DATA/name).read_text())

@lru_cache(None)
def _arrays(name):
    return np.load(DATA/name,allow_pickle=False)

def control(key):
    return _arrays('controls.npz')[key].copy()

def initialization_control(key):
    return _arrays('initialization_controls.npz')[key].copy()
