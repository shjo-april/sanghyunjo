- Install packages to build a wheel file (.whl)
```bash
python -m pip install setuptools wheel --upgrade
```

- Build and install a package (a wheel file)
```bash
python setup.py sdist bdist_wheel
pip install dist/sanghyunjo-1.5.12-py3-none-any.whl --force-reinstall
```

- Release a package in PYPI
```bash
pip install twine
python -m twine upload dist/sanghyunjo-1.5.12*
```