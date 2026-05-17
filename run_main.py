import os
import sys
import streamlit.web.cli as stcli

def resolve_path(path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, path)

if __name__ == "__main__":
    script_path = resolve_path("app_corrected.py_2.py")
    sys.argv = ["streamlit", "run", script_path, "--global.developmentMode=false", "--server.headless=false"]
    sys.exit(stcli.main())
