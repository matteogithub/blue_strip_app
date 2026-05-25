"""
main.py
-------
Entry point for the Blue Strip Depth Analysis GUI.

Run with:
    python main.py

Requirements (all pip-installable):
    numpy scipy matplotlib scikit-image pillow
"""
import sys
import os

# Ensure the app root is on sys.path so all imports resolve correctly
# regardless of where the script is called from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Set matplotlib backend before any other matplotlib import
import matplotlib
matplotlib.use("TkAgg")

from gui.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
