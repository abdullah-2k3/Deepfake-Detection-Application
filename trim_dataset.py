import os
import random
import shutil

SOURCE_DIR = "Dataset"
TRIMMED_DIR = "Trimmed Dataset"
PERCENTAGE_TO_KEEP = 0.1

def stratified_sample_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    num_keep = int(len(files) * PERCENTAGE_TO_KEEP)
    
    # Shuffle and pick
    sampled_files = random.sample(files, num_keep)
    
    for f in sampled_files:
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))

# Traverse directory structure
if __name__ == "__main__":
    if not os.path.exists(TRIMMED_DIR):
        os.makedirs(TRIMMED_DIR)

    splits = ['Train', 'Validation', 'Test']
    classes = ['Fake', 'Real']

    for split in splits:
        src_split_dir = os.path.join(SOURCE_DIR, split)
        dst_split_dir = os.path.join(TRIMMED_DIR, split)

        if not os.path.isdir(src_split_dir):
            continue

        if not os.path.exists(dst_split_dir):
            os.makedirs(dst_split_dir)

        for cls in classes:
            src_cls_dir = os.path.join(src_split_dir, cls)
            dst_cls_dir = os.path.join(dst_split_dir, cls)

            if os.path.isdir(src_cls_dir):
                print(f"Trimming {src_cls_dir} -> {dst_cls_dir}")
                stratified_sample_folder(src_cls_dir, dst_cls_dir)