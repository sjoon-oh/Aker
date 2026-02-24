#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

FILENAME = "spacev-1m-split.npy"
NUM_PARTS = 10

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("-s", "--split", action="store_true")
group.add_argument("-m", "--merge", action="store_true")

args = parser.parse_args()

if args.split:
    data = np.load(FILENAME)
    parts = np.array_split(data, NUM_PARTS)
    for i, part in enumerate(parts):
        out = f"spacev-1m-split.part_{i:02d}.npy"
        np.save(out, part)
        print(f"Saved {out} {part.shape}")

elif args.merge:
    parts = [np.load(f"spacev-1m-split.part_{i:02d}.npy") for i in range(NUM_PARTS)]
    data = np.concatenate(parts, axis=0)
    np.save(FILENAME, data)
    print(f"Merged -> {FILENAME} {data.shape}")