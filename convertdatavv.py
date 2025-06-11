#!/usr/bin/env python3

import lmdb
import pickle
import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def estimate_map_size_bytes(root_dir, buffer_ratio=3.8):
    total = 0
    for dp, _, files in os.walk(root_dir):
        total += sum(
            os.path.getsize(os.path.join(dp, f))
            for f in files if f.lower().endswith(".jpg")
        )
    return int(total * buffer_ratio)

def build_lmdb(
    source_dir: str,
    train_csv:   str,
    lmdb_path:   str,
):
    # 1) CSV → ids & labels
    df = pd.read_csv(train_csv, sep=None, engine="python")
    df = df.drop_duplicates(subset="tomo_id", keep="first")
    ids = df["tomo_id"].tolist()
    labels = df.set_index("tomo_id")["Number of motors"].to_dict()

    # 2) dynamic map_size estimate
    map_size = estimate_map_size_bytes(source_dir)
    print(f"▶️  Estimating map_size = {map_size//(1<<30)} GiB ({map_size} bytes)")

    # 3) open LMDB
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    with env.begin(write=True) as txn:
        for tomo_id in tqdm(ids, desc="LMDB build"):
            folder = os.path.join(source_dir, tomo_id)
            if not os.path.isdir(folder):
                tqdm.write(f"⚠️  missing folder {tomo_id}, skipping")
                continue

            # load & stack
            slices = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
            volume = np.stack([
                np.array(Image.open(os.path.join(folder, f)))
                for f in slices
            ], axis=0)

            record = {"volume": volume, "label": int(labels[tomo_id])}
            data = pickle.dumps(record)

            try:
                txn.put(tomo_id.encode("utf-8"), data)
            except lmdb.MapFullError:
                # **optional** on-the-fly resize
                current = env.info()['map_size']
                new_size = current * 2
                print(f"🔄 Map full – resizing {current} → {new_size} bytes")
                env.set_mapsize(new_size)
                txn.put(tomo_id.encode("utf-8"), data)

    env.sync()
    env.close()
    print(f"✅ Built LMDB at {lmdb_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an LMDB of tomogram volumes + labels"
    )
    parser.add_argument(
        "--source_dir",
        default="byu-locating-bacterial-flagellar-motors-2025/train/",
        help="root folder containing tomo_id subfolders",
    )
    parser.add_argument(
        "--train_csv",
        default="train_labels.csv",
        help="CSV with columns tomo_id, Number of motors",
    )
    parser.add_argument(
        "--lmdb_path",
        default="train.lmdb",
        help="output LMDB file",
    )

    args, _ = parser.parse_known_args()
    build_lmdb(
        source_dir=args.source_dir,
        train_csv=args.train_csv,
        lmdb_path=args.lmdb_path,
    )
#!pip install lmdb pillow numpy tqdm pandas
