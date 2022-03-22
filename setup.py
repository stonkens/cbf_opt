import os
import setuptools

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
    with open(os.path.join(_CURRENT_DIR, "cbf_opt", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `cbf_opt/__init__.py`")


setuptools.setup(
    name="cbf_opt",
    version=_get_version(),
    description="CBF QP implementation in Python",
    author="Sander Tonkens",
    author_email="sandertonkens@gmail.com",
    python_requires=">=3.6",
)
