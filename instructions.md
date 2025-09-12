# Instructions for using pip in venv

## TL;DR
python3.10 -m venv .venv3.10 --prompt py3.10
source .venv3.10/bin/activate
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt

## Overview
Ubuntu 24.04 requires Python to be managed via virtual environments to ensure compatibility and avoid conflicts between system-level packages and project-specific dependencies. In this project, we have chosen to use `venv` to manage Python environments.

## Steps for Installation

1. **Create a Virtual Environment**
   First, create a Python 3.10 virtual environment (we recommend using Python 3.10 because `wrf-python` does not yet support Python 3.11 or higher):
```bash
python3.10 -m venv .venv3.10 --prompt py3.10
```
2. **Activate the Virtual Environment**
   After creating the virtual environment, activate it using:
```bash
source .venv3.10/bin/activate
```
3. **Install pip-tools**
   To manage Python package dependencies more effectively, we'll use pip-tools. Install it by running:
```bash
pip install pip-tools
```
4. **Compile the Dependencies**
   Next, compile the dependencies from the `requirements.in` file to a `requirements.txt` file using:
```bash
pip-compile requirements.in
```
5. **Install the Requirements**
   Install the dependencies listed in the `requirements.txt` by running:
```bash
pip install -r requirements.txt
```
6. **Tests**
   TO-DO run the scripts in a future `tests` folder to ensure correct installation

## Notes
- Python Version Requirement: As of now, `Python` must be version `<=3.10` for compatibility with `wrf-python`. In the near future `wrf-python` will update, and so will we at some point.

- Minimal Pip Instructions: After activating your virtual environment, you can use pip to install or upgrade packages as needed.
   - To install a package: pip install <package-name>
   - To upgrade a package: pip install --upgrade <package-name>
   - To check installed packages: pip list
   - To freeze installed packages to a requirements.txt: pip freeze > requirements.txt

## Warnings
- Avoid upgrading Python: Do not upgrade `Python`. Versions `3.10.X` are the limit
- Avoid upgrading `pip`. `wrf-python` is not fully `pip` compliant, newer `pip` raise errors rather than warnings
- Avoid upgrading `numpy` over `1.26.4`, `MetPy` will fail without warning (it installs, but raise error upon import in `python`)
- Reproducibility: Remember to always use the virtual environment setup in these instructions.
