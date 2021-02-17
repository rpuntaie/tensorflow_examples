#!/usr/bin/env python

"""
Install the packages needed to run the samples.
"""

from setuptools import setup




setup(name='rstdoc',
    version=version,
    description='rstdoc - support documentation in restructedText (rst)',
    license='MIT',
    author='Roland Puntaier',
    keywords=['Documentation'],
    author_email='roland.puntaier@gmail.com',
    url='https://github.com/rstdoc/rstdoc',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Information Technology',
        'Topic :: Utilities',
        ],
    install_requires=['cffi','cairocffi','cairosvg',
                      'pillow', 'pyx', 'pyfca', 'pygal',
                      'numpy', 'matplotlib','sympy','pint>=0.14','drawsvg',
                      'svgwrite', 'stpl>=1.13.6', 'pypandoc', 'docutils',
                      'sphinx', 'sphinx_bootstrap_theme',
                      'gitpython', 'pyyaml','txdir'],
    extras_require={'develop': ['mock', 'virtualenv', 'pytest-coverage'],
                    'build': ['waf']},
    long_description=long_description(),
    packages=['rstdoc'],
    package_data={'rstdoc': ['../readme.rst','reference.tex', 'reference.docx',
                             'reference.odt', 'wafw.py']},
    data_files=[("man/man1", ["rstdoc.1"])],
    zip_safe=False,
    tests_require=['pytest', 'pytest-coverage', 'mock'],
    entry_points={
        'console_scripts': [
            'rstlisttable=rstdoc.listtable:main',
            'rstreflow=rstdoc.reflow:main',
            'rstreimg=rstdoc.reimg:main',
            'rstretable=rstdoc.retable:main',
            'rstdcx=rstdoc.dcx:main',
            'rstdoc=rstdoc.dcx:main',
            'rstfromdocx=rstdoc.fromdocx:main',
            'rstuntable=rstdoc.untable:main',
            ]
    },
)
