Using a `requirements.txt` file is the easiest way to install **all required libraries** at once.
Here’s a **step-by-step beginner-friendly guide** 👇

---

## **1. Make sure you have Python and pip installed**

Open your terminal (Command Prompt, PowerShell, or Mac/Linux terminal) and check:

```bash
python --version
```

or sometimes:

```bash
python3 --version
```

Then check if **pip** (Python’s package manager) is installed:

```bash
pip --version
```

> If it shows a version, you’re good.
> If not, install pip from [https://pip.pypa.io/en/stable/installation/](https://pip.pypa.io/en/stable/installation/).

---

## **2. Create a `requirements.txt` file**

Inside your project folder, create a file named **`requirements.txt`** and put this content inside:

```
numpy
pandas
scikit-learn
matplotlib
scikit-learn-extra   # needed only for K-Medoids
```

---

## **3. Open a terminal in your project folder**

Navigate to the folder where your Python scripts and `requirements.txt` are located:

```bash
cd path/to/your/project
```

Example (Windows):

```bash
cd C:\Users\Sudip\Documents\ClusteringProject
```

Example (Mac/Linux):

```bash
cd ~/Documents/ClusteringProject
```

---

## **4. (Optional but Recommended) Create a Virtual Environment**

This avoids messing up your system Python.

```bash
python -m venv venv
```

Then **activate** it:

* **Windows (Command Prompt):**

  ```bash
  venv\Scripts\activate
  ```
* **Windows (PowerShell):**

  ```bash
  venv\Scripts\Activate.ps1
  ```
* **Mac/Linux:**

  ```bash
  source venv/bin/activate
  ```

You’ll see `(venv)` at the start of your terminal — that means it’s active ✅

---

## **5. Install all libraries from `requirements.txt`**

Once inside your project folder and with your virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This will install:

* **numpy** → for numerical computations
* **pandas** → for handling CSV files
* **scikit-learn** → for K-Means, MiniBatch, DBSCAN, Agglomerative, etc.
* **matplotlib** → for plotting graphs
* **scikit-learn-extra** → for **K-Medoids**

---

## **6. Verify installation**

After installation, you can check if all are installed:

```bash
pip list
```

You should see **numpy**, **pandas**, **scikit-learn**, **matplotlib**, and **scikit-learn-extra**.

---

## **7. Run your scripts**

Now you can run any of your clustering scripts. For example:

```bash
python kmeans_sample.py
```

or, if you’re on Mac/Linux:

```bash
python3 kmeans_sample.py
```

---

## **8. If you get any issues**

* **Upgrade pip** (sometimes fixes installation problems):

  ```bash
  python -m pip install --upgrade pip
  ```
* **Install scikit-learn-extra separately** if needed:

  ```bash
  pip install scikit-learn-extra
  ```

---

## **Summary**

* Put all libraries in `requirements.txt`
* Run:

  ```bash
  pip install -r requirements.txt
  ```
* Done ✅

---

Do you also want me to create **one universal Python script** that lets you **choose the clustering algorithm** (**K-Means, MiniBatch, K-Medoids, DBSCAN, Agglomerative**) **and plots automatically**?
It’ll make testing all algorithms on the same dataset super easy. Should I?




/// **yo her**
# Step 1 — Create venv (optional but recommended)
python -m venv venv

# Step 2 — Activate venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Step 3 — Install all libraries (needs internet ONCE)
pip install -r requirements.txt

# Step 4 — Run any script (no internet needed anymore)
python kmeans_sample.py
