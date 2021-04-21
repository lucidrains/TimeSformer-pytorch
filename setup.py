from setuptools import setup, find_packages

setup(
  name = 'timesformer-pytorch',
  packages = find_packages(),
  version = '0.2.1',
  license='MIT',
  description = 'TimeSformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/TimeSformer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'video classification',
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
