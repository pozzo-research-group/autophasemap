from setuptools import setup,find_packages
import sys, os

setup(name="autophasemap",
      description="Automatic Structure Phase Mapping from Functional Data",
      version='1.0',
      author='Kiran Vaddi',
      author_email='kiranvad@uw.edu',
      license='MIT',
      python_requires='>=3.8',
      install_requires=['numpy>=1.18.1','scipy', 'matplotlib', 
      'Cython==0.29.30', 'cffi==1.15.0', 'pygsp==0.5.1', 'ortools>=9.4.1874'],
      extras_require = {},
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows"
      ],
)