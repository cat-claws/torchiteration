from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.0.5'

requirements = [
    'numpy',
    'pandas',
]

setup(
    # Metadata
    name='torchiteration',
    version=VERSION,
    author='cat-claws',
    author_email='47313357+cat-claws@users.noreply.github.com',
    url='https://github.com/cat-claws/torchiteration',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
