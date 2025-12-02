from setuptools import setup, find_packages

setup(
    name='sinusRobot',
    version='0.1.0',
    author='SimonB111',
    description='Sinus robot visualization and calibration',
    packages=find_packages(),
    install_requires=[
        'pyvista',
        'pyvistaqt',
        'PyQt5',
        'opencv-python',
        'numpy',
        'scipy',
    ],
    python_requires='>=3.8',
)
