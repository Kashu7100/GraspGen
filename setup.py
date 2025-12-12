import os
import os.path as osp
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Helper to get the current directory
this_dir = osp.dirname(osp.abspath(__file__))

# PointNet2 Ops Extension Configuration
_ext_src_root = osp.join("grasp_gen", "pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)

# Only build extension if CUDA is available or if we force it?
# Usually CUDAExtension handles the check or fails.
# Since this library heavily relies on it, we assume it's needed.

ext_modules = [
    CUDAExtension(
        name="grasp_gen.pointnet2_ops._ext",
        sources=_ext_sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
        },
        include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
    )
]

install_requires = []

setup(
    name="eden_grasp_gen",
    version="0.1.2",
    packages=find_packages(),
    package_data={
        "grasp_gen": ["config/grippers/*.yaml", "config/grippers/*.py", "assets/**/*"],
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    description="A fork of GraspGen for EDEN project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kashu Yamazaki",
    author_email="kyamazak@andrew.cmu.edu",
    license="NVIDIA License",
    url="https://github.com/NVlabs/GraspGen",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.10",
)
