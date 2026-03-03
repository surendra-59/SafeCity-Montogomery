
---

# 📝 Quick Start Guide

> **Note:** Anything inside
>
> ```
> ```
>
> is a command you run in your terminal.

---

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

---

### 2. Activate Virtual Environment

```bash
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
uv sync
```

> This reads `pyproject.toml` and installs all required packages into a virtual environment.

---


### 4. Run Jupyter Lab (optional)

```bash
uv run jupyter lab
```

> Use this if you want to explore or modify notebooks.

---

### 5. Run the application

To start the Streamlit app:

```bash
uv run streamlit run dashboard.py
```

---
