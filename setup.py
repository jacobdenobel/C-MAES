import os
import subprocess
import platform
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
    "c_maes.cmaescpp", 
    glob("src/*cpp"), 
    include_dirs=[
        "include",
        "external"
    ],
    cxx_std=17
)
if platform.system() in ("Linux", "Darwin"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
    ext._add_cflags(["-O3"])
else:
    ext._add_cflags(["/O2"])


with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    description = f.read()
    
setup(
    name="c_maes",
    author="Jacob de Nobel",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    description="C++ Implementation of Modular CMA-ES",
    long_description=description,
    zip_safe=False,
    version = "0.0.1",
    install_requires = [
        "matplotlib>=3.3.4",
        "numpy>=1.19.2",
        "scipy>=1.5.2"
    ]
)