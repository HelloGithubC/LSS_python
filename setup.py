import os
import sys
import subprocess

# ========== 在 setup() 之前构建 C++ 库 ==========
def build_cpp_extensions():
    if not os.path.exists("src/LSS_python/CPP/lib"):
        os.makedirs("src/LSS_python/CPP/lib", exist_ok=True)
    
    print("Building C++ extensions with Makefile...")
    result = subprocess.run(
        ["make"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print("Make failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("C++ extensions built successfully")

# 关键：在 setup() 被调用前执行！
build_cpp_extensions()
# ============================================

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

class CustomInstall(install):
    def run(self):
        super().run()

class CustomDevelop(develop):
    def run(self):
        super().run()

setup(
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    }
)