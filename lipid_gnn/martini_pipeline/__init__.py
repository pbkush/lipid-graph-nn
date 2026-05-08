import os

_PACKAGE_DIR = os.path.dirname(__file__)

INSANE_PATH: str = os.path.normpath(
    os.path.join(_PACKAGE_DIR, "..", "..", "resources", "martini3", "insane.py")
)
