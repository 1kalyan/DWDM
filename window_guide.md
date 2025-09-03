Got you. Here’s a **clear, step-by-step Windows guide** for BOTH ways:

* **A) Using a virtual environment (venv)** — recommended
* **B) Installing globally (without venv)** — OK for quick tests

I’ll show commands for **Command Prompt (CMD)** and **PowerShell** where they differ.

---

# A) Using a virtual environment (venv) ✅ (recommended)

## 0) Open a terminal in your project folder

```bat
cd C:\path\to\your\project
```

*(Right-click in the folder → “Open in Terminal” works too.)*

## 1) Check Python & pip

```bat
python --version
python -m pip --version
```

> If these fail, install Python from python.org and check “Add Python to PATH”.

## 2) Create a venv

```bat
python -m venv venv
```

This makes a folder `venv/` in your project.

## 3) Activate the venv

* **CMD:**

  ```bat
  venv\Scripts\activate
  ```
* **PowerShell:**

  ```powershell
  venv\Scripts\Activate.ps1
  ```

You should now see `(venv)` at the start of the prompt.

> If PowerShell blocks activation, run once:
>
> ```powershell
> Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## 4) Upgrade pip, setuptools, wheel (inside venv)

```bat
python -m pip install --upgrade pip setuptools wheel
```

## 5) Install your libraries

If you have `requirements.txt`:

```bat
pip install -r requirements.txt
```

*(Use the pinned file we made earlier, e.g. with numpy==1.26.4, etc.)*

Or install individually:

```bat
pip install numpy pandas scikit-learn matplotlib scikit-learn-extra mlxtend seaborn
```

## 6) Verify

```bat
pip list
python -c "import numpy, pandas, sklearn, matplotlib; print('OK')"
```

## 7) Run your scripts

```bat
python kmeans_sample.py
```

## 8) Deactivate (when you’re done)

```bat
deactivate
```

*(To “reset” the env, delete the `venv/` folder and re-create it.)*

---

# B) Installing **without** venv (global install)

> Use this for quick/one-off work. It affects your whole system Python.

## 1) Open CMD/PowerShell and check Python

```bat
python --version
```

## 2) Upgrade pip (global)

```bat
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install libs globally

With requirements:

```bat
pip install -r requirements.txt
```

Or individually:

```bat
pip install numpy pandas scikit-learn matplotlib scikit-learn-extra mlxtend seaborn
```

## 4) Verify

```bat
pip list
python -c "import numpy, pandas, sklearn, matplotlib; print('OK')"
```

---

## Quick tips & common fixes

* **Use `python -m pip ...`** instead of `pip ...` when in doubt (avoids version/path mixups).
* **If `pip` not recognized:**

  ```bat
  python -m ensurepip --upgrade
  ```
* **NumPy 2.x vs compiled packages:** if you hit binary-compatibility errors (e.g., with `scikit-learn-extra`), pin versions in `requirements.txt` (e.g., `numpy==1.26.4`) and reinstall.
* **PowerShell policy error on activate:**
  `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
* **Stay consistent:** Always **activate the venv** before installing/running so packages go into the right place.

---

## Minimal command set (copy–paste)

### With venv

```bat
cd C:\path\to\your\project
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python your_script.py
```

### Without venv

```bat
cd C:\path\to\your\project
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python your_script.py
```

If you want, I can also give you a **Windows `setup.bat`** that automates: create venv → activate → upgrade pip → install requirements.
