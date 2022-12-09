from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fused',
      ext_modules=[cpp_extension.CppExtension('fused', ['/ocean/projects/asc170022p/singla/CounterfactualExplainer/MIMICCX-Chest-Explainer/stylegan2Pytorch/op/fused_bias_act.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})