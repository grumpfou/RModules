from setuptools import setup, find_packages
import platform



setup(name='mymodules',
      version='0.0',
      description='Miscelanous modules for me',
      long_description="""This file contains several miscelanous modules for me
      """,
      classifiers=[
        'Development Status :: 0.0',
        'License :: GPL-3.0',
        'Programming Language :: Python :: 3.7',
      ],
      python_requires='>=3.7',
      # url='https://github.com/grumpfou/RFigure',
      # author='Renaud Dessalles',
      # author_email='see on my website',
      license='GPL-3.0',
      py_modules=['RMatplotlib','RVideoCreator'],
      # include_package_data=True,
      # install_requires=[],
      )
