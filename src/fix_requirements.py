import os

def clean_requirements():
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"[ERROR] {req_file} not found.")
        return
        
    print(f"[INFO] Cleaning {req_file} for Streamlit Cloud deployment...")
    try:
        with open(req_file, "r", encoding="utf-16") as f:
            lines = f.readlines()
    except UnicodeError:
        with open(req_file, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
            
        # 1. Skip Windows-only packages
        if stripped.startswith("pywin32"):
            print(f"  Removing Windows-only package: {stripped}")
            continue
            
        # 2. Convert opencv-python to opencv-python-headless
        if stripped.startswith("opencv-python=="):
            print(f"  Replacing {stripped} with opencv-python-headless")
            cleaned_lines.append("opencv-python-headless" + stripped[len("opencv-python"):] + "\n")
            continue
            
        # 3. Clean torch CPU suffixes
        if stripped.startswith("torch=="):
            # Replace torch==2.12.1+cpu with standard torch version or simply torch>=2.0.0
            print(f"  Replacing {stripped} with torch")
            cleaned_lines.append("torch\n")
            continue
            
        if stripped.startswith("torchvision=="):
            print(f"  Replacing {stripped} with torchvision")
            cleaned_lines.append("torchvision\n")
            continue
            
        cleaned_lines.append(line)
        
    with open(req_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)
        
    print("[SUCCESS] requirements.txt has been successfully cleaned for Linux compatibility.")

if __name__ == "__main__":
    clean_requirements()
