import os

_PACKAGE_DIR = os.path.dirname(__file__)

INSANE_CMD: str = "insane"

MARTINI3_ITP_DIR: str = os.path.normpath(
    os.path.join(_PACKAGE_DIR, "..", "..", "resources", "martini3", "itp")
)
