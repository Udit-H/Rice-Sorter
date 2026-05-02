import os
import shutil
import random

SOURCE_DIR = "./rice_dataset_files/rice_extracted"   # e.g. "rice_images"
TARGET_DIR = "dataset"      # output folder

TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1

random.seed(42)  # reproducibility

def split_dataset():
    # iterate over each rice type subdirectory
    for cls in os.listdir(SOURCE_DIR):
        src_cls_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        # collect all image files
        images = [f for f in os.listdir(src_cls_dir) 
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(TRAIN_SPLIT * n_total)
        n_val   = int(VAL_SPLIT * n_total)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train+n_val],
            "test":  images[n_train+n_val:]
        }

        # copy into target structure
        for split, files in splits.items():
            split_dir = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_cls_dir, f),
                            os.path.join(split_dir, f))

        print(f"{cls}: total={n_total}, train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

if __name__ == "__main__":
    split_dataset()
