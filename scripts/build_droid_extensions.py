"""Build TRAM's vendored masked DROID for the local CUDA architecture only.

Run from thirdparty/DROID-SLAM with CUDA_HOME pointing at an isolated toolkit.
No system CUDA or other conda environment is modified.
"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parents[1] / 'thirdparty' / 'DROID-SLAM'
os.chdir(root)
assert (root / 'src/droid.cpp').is_file(), f'Missing DROID source: {root}'
arch = os.environ.get('TRAM_CUDA_ARCH', '86')
flags = {'cxx': ['-O2'], 'nvcc': ['-O2', f'-gencode=arch=compute_{arch},code=sm_{arch}']}
setup(name='tram-droid-extensions', version='0.1',
      packages=['lietorch'], package_dir={'': 'thirdparty/lietorch'},
      ext_modules=[
          CUDAExtension('droid_backends',
                        include_dirs=[str(root/'thirdparty/eigen')],
                        sources=['src/droid.cpp', 'src/droid_kernels.cu',
                                 'src/correlation_kernels.cu', 'src/altcorr_kernel.cu'],
                        extra_compile_args=flags),
          CUDAExtension('lietorch_backends',
                        include_dirs=[str(root/'thirdparty/lietorch/lietorch/include'),
                                      str(root/'thirdparty/eigen')],
                        sources=['thirdparty/lietorch/lietorch/src/lietorch.cpp',
                                 'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                                 'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
                        extra_compile_args=flags)],
      cmdclass={'build_ext': BuildExtension})
