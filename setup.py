from setuptools import setup, find_packages

setup(name='random-envs',
      version='0.0.1',
      install_requires=['gym', 'jinja2', 'scipy', 'mujoco_py', 'patchelf'],
      packages=find_packages(exclude=["deprecated_tests"]),
      package_data={'random_envs': ['jinja/assets/*.xml']},
      include_package_data=True,
)