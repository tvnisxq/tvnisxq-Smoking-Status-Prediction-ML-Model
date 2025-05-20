import os
import sys

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Print Python path
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

# Find all Python files in current directory and 2 levels up
print("\nSearching for Python files...")

def find_python_files(start_dir, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return
    
    try:
        for item in os.listdir(start_dir):
            path = os.path.join(start_dir, item)
            if os.path.isfile(path) and path.endswith('.py'):
                print(f"  {path}")
            if os.path.isdir(path) and not item.startswith('.'):
                find_python_files(path, max_depth, current_depth + 1)
    except PermissionError:
        print(f"  Permission denied for {start_dir}")
    except Exception as e:
        print(f"  Error accessing {start_dir}: {e}")

# Start search from current directory
current_dir = os.getcwd()
print(f"\nPython files in current directory and subdirectories:")
find_python_files(current_dir)

# Also search parent directory
parent_dir = os.path.dirname(current_dir)
print(f"\nPython files in parent directory and subdirectories:")
find_python_files(parent_dir)

# Try to locate the 'src' directory specifically
print("\nLooking for 'src' directory:")
for root, dirs, files in os.walk(parent_dir):
    if 'src' in dirs:
        src_path = os.path.join(root, 'src')
        print(f"  Found src directory at: {src_path}")
        print("  Contents:")
        for item in os.listdir(src_path):
            print(f"    {item}")

input("\nPress Enter to exit...")