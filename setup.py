from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys, glob
import os

def build_cpp_extensions():
    """构建 C++ 共享库"""
    print("Building C++ extensions with Makefile...")
    result = subprocess.run(
        ['make'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print("Make failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    print("C++ extensions built successfully")

class CustomInstall(install):
    def run(self):
        build_cpp_extensions()
        super().run()

class CustomDevelop(develop):
    def run(self):
        build_cpp_extensions()
        super().run()

setup(
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    }
)