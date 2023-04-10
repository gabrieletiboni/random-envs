from setuptools import setup, find_packages

setup(name='random_envs',
      version='0.0.1',
      packages=find_packages(exclude=["deprecated_tests"]),
      package_data={'random_envs': ['jinja/assets/*.xml']},
      include_package_data=True,
)