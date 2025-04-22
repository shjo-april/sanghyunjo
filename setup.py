from setuptools import setup, find_packages

import sanghyunjo as shj

setup(
    name='sanghyunjo',
    version=shj.__version__,
    description='Wrapped utility functions for existing AI packages to simplify their usage',
    author='Sanghyun Jo (shjo-april)',
    author_email='shjo.april@gmail.com',
    url='https://github.com/shjo-april/sanghyunjo',
    install_requires=[
        'joblib',
        'cmapy',
        'tqdm',
        'opencv-python',
        'Pillow',
        'requests',
        
        # 'scikit-learn' and 'torch' are optional dependencies (e.g., for GPU or ML features)
        # 'scikit-learn',
        # 'torch'
    ],
    packages=find_packages(exclude=[]),
    keywords=['sanghyun', 'sanghyunjo', 'shjo', 'ai', 'utils', 'wrapper'],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        # Include the font file in the package
        'sanghyunjo': ['fonts/Times New Roman MT Std.otf'],
    },
    data_files=[
        ('fonts', ['sanghyunjo/fonts/Times New Roman MT Std.otf']),
    ],
    zip_safe=False,
)
