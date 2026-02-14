
import py_compile
import sys

print("Checking syntax of src/veldra/gui/app.py...")
try:
    py_compile.compile('src/veldra/gui/app.py', doraise=True)
    print("Syntax OK")
except Exception as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)
