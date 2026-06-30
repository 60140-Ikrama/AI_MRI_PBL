import os
import shutil

def organize_dataset():
    data_dir = "data"
    brisc_dir = os.path.join(data_dir, "brisc2025")
    seg_task_dir = os.path.join(brisc_dir, "segmentation_task")
    
    if not os.path.exists(seg_task_dir):
        print(f"Error: {seg_task_dir} does not exist. Make sure the dataset is fully downloaded and extracted.")
        return

    # Clean existing data/train and data/test directories if they exist
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)

    # Move files from segmentation_task to data/train and data/test
    for split in ["train", "test"]:
        src_img_dir = os.path.join(seg_task_dir, split, "images")
        src_mask_dir = os.path.join(seg_task_dir, split, "masks")
        
        dest_img_dir = os.path.join(data_dir, split, "images")
        dest_mask_dir = os.path.join(data_dir, split, "masks")
        
        print(f"Organizing {split} split...")
        
        # Move images
        if os.path.exists(src_img_dir):
            for file_name in os.listdir(src_img_dir):
                shutil.move(os.path.join(src_img_dir, file_name), os.path.join(dest_img_dir, file_name))
        
        # Move masks
        if os.path.exists(src_mask_dir):
            for file_name in os.listdir(src_mask_dir):
                shutil.move(os.path.join(src_mask_dir, file_name), os.path.join(dest_mask_dir, file_name))
                
    # Clean up temporary folders
    print("Cleaning up temporary folders...")
    try:
        if os.path.exists(brisc_dir):
            shutil.rmtree(brisc_dir)
        print("Cleanup complete!")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    organize_dataset()
