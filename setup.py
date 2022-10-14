from distutils.core import setup
from setuptools import find_packages

setup(name='NLP_task_ASU',
      version='1.0',
      description='NLP_task_ASU',
      author='SorokaD',
      author_email='sorokadn@list.ru',
      url='https://github.com/SorokaD/NLP_task_ASU',
      package_dir={"": "src"},
      packages=find_packages(where="src"),
     )
