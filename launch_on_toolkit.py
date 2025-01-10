import os
import glob
import subprocess

def launch_all_scripts_in_directory(directory: str):
    """Finds all .sh files in 'directory' and calls dispatch_toolkit.py to launch each one."""
    sh_files = glob.glob(os.path.join(directory, "*.sh"))
    for script in sh_files:
        # Call dispatch_toolkit with the script path
        cmd = f"python dispatch_toolkit.py --path '{script}'"
        print(f"Launching: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    # Example usage:
    directory_with_scripts = "scripts/flair2_rgb"
    launch_all_scripts_in_directory(directory_with_scripts)