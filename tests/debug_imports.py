import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")

try:
    import dash
    print("Dash imported successfully.")
except ImportError as e:
    print(f"Error importing dash: {e}")

try:
    import dash_bootstrap_components
    print("DBC imported successfully.")
except ImportError as e:
    print(f"Error importing dbc: {e}")

try:
    import veldra
    print("Veldra imported successfully.")
except ImportError as e:
    print(f"Error importing veldra: {e}")

try:
    from veldra.gui.app import create_app
    print("create_app imported successfully.")
except ImportError as e:
    print(f"Error importing create_app: {e}")
    import traceback
    traceback.print_exc()

try:
    from veldra.gui.pages import data_page, results_page
    print("Pages imported successfully.")
except ImportError as e:
    print(f"Error importing pages: {e}")
    import traceback
    traceback.print_exc()
