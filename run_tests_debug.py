
import pytest
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print("Running pytest...")

try:
    ret = pytest.main(["-v", "tests/test_gui_new_layout.py"])
    print(f"Pytest return code: {ret}")
    sys.exit(ret)
except Exception as e:
    print(f"Error running pytest: {e}")
    sys.exit(1)
