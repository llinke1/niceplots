from setuptools import setup

setup(
    name='niceplots',
    version='0.1.0',
    description=' Code for creating nice matplotlib plots',
    url='',
    author='Laila Linke',
    author_email='laila.linke@uibk.ac.at',
    packages=['niceplots'],
    install_requires=['matplotlib'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    include_package_data=True
)