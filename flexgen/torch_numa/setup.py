from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os

# 获取 PyTorch 库路径
torch_lib_path = os.path.join(torch.__path__[0], 'lib')
torch_include_path = os.path.join(torch.__path__[0], 'include')

setup(
    name='torch_numa',
    version='0.0.1',
    packages=find_packages(),  # 使用包结构而不是单独的模块
    ext_modules=[
        CppExtension(
            name='torch_numa_cpp',
            sources=['torch_numa.cpp'],
            include_dirs=[torch_include_path],
            libraries=['numa', 'torch', 'torch_cpu', 'c10'],
            library_dirs=[torch_lib_path],
            extra_compile_args=['-O3', '-std=c++17'],
            extra_link_args=['-lnuma', f'-Wl,-rpath,{torch_lib_path}'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
    ],
    description='PyTorch NUMA memory allocator extension',
    author='PyTorch NUMA Extension Team',
    license='MIT',
)