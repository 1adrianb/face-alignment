from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

requirements = [
    'torch',
    'dlib>=19.5',
    'numpy',
    'scipy>=0.17.0',
    'opencv-python',
    'scikit-image',
    'enum34;python_version<"3.4"'
]

setup(
    name='face_alignment',
    version='0.1.1',

    description="Detector 2D or 3D face landmarks from Python",
    long_description=long_description,

    # Author details
    author="Adrian Bulat",
    author_email="adrian.bulat@nottingham.ac.uk",
    url="https://github.com/1adrianb/face-alignment",

    # Package info
    packages=find_packages(exclude=('test',)),

    install_requires=requirements,
    license='BSD',
    zip_safe=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
