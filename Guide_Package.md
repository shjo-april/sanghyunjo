- Install packages to build a wheel file (.whl)
```bash
python -m pip install setuptools wheel --upgrade
```

- Build a package (a wheel file)
```bash
python setup.py sdist bdist_wheel
```

- Install a package
```bash
pip install dist/sanghyunjo-1.3.0-py3-none-any.whl --force-reinstall
```

- Release a package in PYPI
```bash
pip install twine
python -m twine upload dist/sanghyunjo-1.3.0*
```