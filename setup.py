from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import sys, glob
import os

class CustomBuildExt(build_ext):
    """自定义构建扩展类，用于调用Makefile编译C++库"""
    
    def run(self):
        # 首先运行Makefile编译C++共享库
        print("Building C++ extensions with Makefile...")
        
        # 运行make命令
        result = subprocess.run(
            ['make'], 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))  # 在setup.py所在目录运行
        )
        
        if result.returncode != 0:
            print("Make failed with output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(result.returncode)
        else:
            so_files = glob.glob("src/LSS_python/CPP/lib/*.so")
            if not so_files:
                raise RuntimeError("No .so files found after make! Check Makefile output path.")
            print("C++ extensions built successfully")
            if result.stdout:
                print(result.stdout)

# 由于你通过Makefile直接编译.so文件，我们不需要定义传统的Extension
# 但需要确保.so文件被包含在包中
# setup(
#     name="LSS_python",
#     version="0.5.0",
    
#     # 包发现配置
#     package_dir={"": "src"},
#     packages=find_packages(where="src"),
    
#     # 包含共享库文件
#     package_data={
#         'LSS_python.CPP': ['lib/*.so'],
#     },
    
#     # 确保包中包含.so文件
#     include_package_data=True,
    
#     # 使用自定义构建命令
#     cmdclass={
#         'build_ext': CustomBuildExt,
#     },
    
#     # 安装要求
#     install_requires=[
#         "numpy",
#         "numba==0.61.0",
#         "joblib",
#         "scipy",
#         "emcee",
#         "h5py",
#         "multiprocess",
#         "Corrfunc",
#     ],
    
#     # 其他元数据
#     author="Liang Xiao",
#     author_email="xiaoliangms@outlook.com",
#     description="A package to deal with some problems about Large Scale Structure. Full support by python(numba). cuda(cupy) is optinonal",
#     python_requires=">=3.6",
#     zip_safe=False,  # 由于包含共享库，设置为False
# )

setup(
    cmdclass={"build_ext": CustomBuildExt}
)