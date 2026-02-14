import os
import sys
from importlib import import_module

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")

try:
    dash_mod = import_module("dash")

    print(f"Dash imported successfully: {dash_mod.__name__}")
except ImportError as e:
    print(f"Error importing dash: {e}")

try:
    dbc_mod = import_module("dash_bootstrap_components")

    print(f"DBC imported successfully: {dbc_mod.__name__}")
except ImportError as e:
    print(f"Error importing dbc: {e}")

try:
    veldra_mod = import_module("veldra")

    print(f"Veldra imported successfully: {veldra_mod.__name__}")
except ImportError as e:
    print(f"Error importing veldra: {e}")

try:
    create_app = import_module("veldra.gui.app").create_app

    print(f"create_app imported successfully: {create_app.__name__}")
except ImportError as e:
    print(f"Error importing create_app: {e}")
    import traceback

    traceback.print_exc()

try:
    data_page = import_module("veldra.gui.pages.data_page")
    results_page = import_module("veldra.gui.pages.results_page")

    print(f"Pages imported successfully: {data_page.__name__}, {results_page.__name__}")
except ImportError as e:
    print(f"Error importing pages: {e}")
    import traceback

    traceback.print_exc()
