from setuptools import setup, find_packages

setup(
    name='ml-stress-prediction-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A machine learning application for stress prediction and fringe order estimation.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',  # or 'torch' depending on the model framework
        'opencv-python',
        'streamlit',
        'Pillow',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)