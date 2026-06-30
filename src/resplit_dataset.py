import os
import random
import shutil

def resplit_dataset(train_ratio=0.7):
    data_dir = "data"
    train_img_dir = os.path.join(data_dir, "train", "images")
    train_mask_dir = os.path.join(data_dir, "train", "masks")
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_mask_dir = os.path.join(data_dir, "test", "masks")

    # 1. Pool all files into train first
    print("Pooling all files into train folder...")
    test_images = os.listdir(test_img_dir)
    for img_name in test_images:
        # Move image
        shutil.move(os.path.join(test_img_dir, img_name), os.path.join(train_img_dir, img_name))
        # Move mask (same name, .png extension)
        base_name, _ = os.path.splitext(img_name)
        mask_name = base_name + ".png"
        shutil.move(os.path.join(test_mask_dir, mask_name), os.path.join(train_mask_dir, mask_name))

    # 2. Gather all pooled files from train
    all_images = sorted(os.listdir(train_img_dir))
    total_files = len(all_images)
    print(f"Total files pooled: {total_files}")

    # 3. Calculate split size
    num_train = int(total_files * train_ratio)
    num_test = total_files - num_train
    print(f"Target split - Train: {num_train} (70%), Test: {num_test} (30%)")

    # 4. Shuffle deterministically and pick test files
    random.seed(42)
    test_selection = random.sample(all_images, num_test)

    # 5. Move selected files to test folder
    print(f"Moving {num_test} files to test split...")
    for img_name in test_selection:
        # Move image
        shutil.move(os.path.join(train_img_dir, img_name), os.path.join(test_img_dir, img_name))
        # Move mask
        base_name, _ = os.path.splitext(img_name)
        mask_name = base_name + ".png"
        shutil.move(os.path.join(train_mask_dir, mask_name), os.path.join(test_mask_dir, mask_name))

    print(f"Successfully split dataset: {len(os.listdir(train_img_dir))} train, {len(os.listdir(test_img_dir))} test.")

if __name__ == "__main__":
    resplit_dataset()
