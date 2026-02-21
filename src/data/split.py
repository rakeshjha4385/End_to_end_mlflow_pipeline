import os
import shutil
import random

DATA_DIR = "data/processed"
OUTPUT_DIR = "data/split"

def split_data(train_ratio=0.8, val_ratio=0.1):

    for label in ["cats", "dogs"]:

        files = os.listdir(os.path.join(DATA_DIR, label))
        random.shuffle(files)

        train_end = int(len(files) * train_ratio)
        val_end = int(len(files) * (train_ratio + val_ratio))

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, split_files in splits.items():
            split_path = os.path.join(OUTPUT_DIR, split, label)
            os.makedirs(split_path, exist_ok=True)

            for f in split_files:
                shutil.copy(
                    os.path.join(DATA_DIR, label, f),
                    os.path.join(split_path, f)
                )

    print("Dataset Split Completed")

if __name__ == "__main__":
    split_data()
