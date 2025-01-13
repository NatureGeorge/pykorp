# @Created Date: 2025-01-07 04:05:19 pm
# @Filename: setup.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-01-13 03:53:34 pm
from setuptools import setup, Extension
import pybind11


ext_modules = [
    Extension(
        "pykorp.c_korp",
        sources=["pykorp/interface.cpp"],
        include_dirs=[pybind11.get_include(), "pykorp/include"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="pykorp",
    version="1.0",
    description="Python/Pytorch bindings for KORP.",
    ext_modules=ext_modules,
    license="MIT",
    author_email="zhuzefeng@stu.pku.edu.cn",
)
