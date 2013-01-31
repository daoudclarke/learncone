from setuptools import setup
from disttest import test

# Import this to prevent spurious error: info('process shutting down')
from multiprocessing import util

setup(name='learncone',
      version='0.1',
      description='Machine learning classification using cones for SciKit learn',
      #url='http://github.com/storborg/funniest',
      author='Daoud Clarke',
      #author_email='flyingcircus@example.com',
      license='MIT',
      packages=['learncone'],
      cmdclass = {'test': test},
      options = {'test' : {'test_dir':['test']}}
      #zip_safe=False
      )
