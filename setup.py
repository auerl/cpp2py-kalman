try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    print("Warning: Did not find setuptools!")
    from distutils.core import setup
    from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

import os
submodule_name = "./kalman_filter"
if not os.path.exists(submodule_name):
    os.makedirs(submodule_name)
    
kalman = Extension(
    'kalman',
    sources = ['./kalman.cpp'],
    extra_compile_args = ["-DARMA_DONT_USE_WRAPPER -lopenblas -llapack"],
    libraries=['openblas', 'lapack']
)

setup(
    ext_modules=[kalman],
    package_dir={'kalman': ''},
    include_dirs=[np.get_include(), './']
)

for file in os.listdir(submodule_name):
    os.rename("{}/{}".format(submodule_name,file), file)
os.rmdir(submodule_name)
