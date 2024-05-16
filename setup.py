from setuptools import setup, find_packages

setup(
    name='sanghyunjo',
    version='1.0.0',
    description='Wrapped utility functions for existing AI packages to simplify their usage',
    author='Sanghyun Jo (shjo-april)',
    author_email='shjo.april@gmail.com',
    url='https://github.com/shjo-april/sanghyunjo',
    install_requires=[
        'json', 
        'opencv-python', 'cmapy', 'numpy', 'Pillow',
        'tqdm', 'joblib'
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
