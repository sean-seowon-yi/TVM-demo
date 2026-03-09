import os, glob, subprocess
# Search broadly
for root in [os.path.expanduser("~/tvm_env")]:
    for p in glob.glob(os.path.join(root, "**", "nvcc"), recursive=True):
        print(f"Found: {p}  exec={os.access(p, os.X_OK)}")
    for p in glob.glob(os.path.join(root, "**", "nvcc.bin"), recursive=True):
        print(f"Found: {p}  exec={os.access(p, os.X_OK)}")

# Check nvidia package location
try:
    result = subprocess.run(
        [os.path.expanduser("~/tvm_env/bin/pip"), "show", "nvidia-cuda-nvcc-cu12"],
        capture_output=True, text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"pip show failed: {e}")
