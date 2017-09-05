
# from setuptools import find_packages
# from setuptools import setup

# REQUIRED_PACKAGES = ['numpy>=1.12.0', 'fitsio >= 0.9.11']

# setup(
#     name='fitsio',
#     version='0.9.12rc1',
#     install_requires=REQUIRED_PACKAGES,
#     packages=find_packages(),
#     include_package_data=True,
#     description='My trainer application package.'
# )

from setuptools import setup, find_packages

setup(name='example5',
  version='0.1',
  packages=find_packages(),
  description='example to run keras on gcloud ml-engine',
  author='Fuyang Liu',
  author_email='fuyang.liu@example.com',
  license='MIT',
  install_requires=[
      'keras',
      'fitsio',
      'h5py'
  ],
  zip_safe=False)
