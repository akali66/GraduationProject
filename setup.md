# Installation & Setup on Windows 11

## 1. Prerequisites and Python Version Recommendation

**We recommend Python 3.12.**

*   **Why Python 3.12?**: It offers excellent performance improvements (specializing adaptive interpreter) and robust type hinting, while retaining full compatibility with pre-built binary wheels for major scientific libraries (OpenCV, NumPy, Matplotlib) on Windows 11. 
*   **Avoid Source Builds**: `pip` will automatically download the pre-compiled `.whl` files for the versions specified in `requirements.txt`, avoiding the need for visual studio build tools (C++ compilers).

## 2. Environment Setup Steps

### 2.1 Option A: Using Command Prompt (`cmd.exe`)

```cmd
:: 1. Navigate to your project directory
cd path\to\project

:: 2. Create the virtual environment
python -m venv .venv

:: 3. Activate the virtual environment
.venv\Scripts\activate.bat

:: 4. Upgrade pip (recommended)
python -m pip install --upgrade pip

:: 5. Install the required dependencies
pip install -r requirements.txt
```

### 2.2 Option B: Using PowerShell (`powershell.exe` or `pwsh.exe`)

```powershell
# 1. Navigate to your project directory
cd path\to\project

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Note: You might need to adjust execution policies the first time (see Troubleshooting)
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip 
python -m pip install --upgrade pip

# 5. Install the required dependencies
pip install -r requirements.txt
```

## 3. Troubleshooting Windows Issues

### 3.1 Cannot Activate Virtual Environment (PowerShell `ExecutionPolicy` error)
*   **Error**: `cannot be loaded because running scripts is disabled on this system.`
*   **Fix**: You need to allow the running of downloaded scripts in your current session. Open an administrative PowerShell, or run this in your normal PowerShell window before activating:
    ```powershell
    Set-ExecutionPolicy Unrestricted -Scope CurrentUser
    ```

### 3.2 OpenCV Build/Installation Errors
*   **Symptom**: `pip` attempts to build `opencv-python` from source and fails with C++ compilation errors.
*   **Fix**: This means a pre-built wheel (binary) wasn't available for your CPU structure (e.g., ARM64 Windows) or specific Python sub-version. Ensure you are using a standard 64-bit (x64) Windows install of Python 3.12. Alternatively, try upgrading `pip` before installing: `python -m pip install --upgrade pip setuptools wheel`. The pinned version `4.9.0.80` has x64 binary wheels uploaded.

### 3.3 Missing Tkinter
*   **Symptom**: `ModuleNotFoundError: No module named 'tkinter'`
*   **Fix**: Tkinter is part of the standard Python library on Windows. However, during the Python Windows Installer process, "tcl/tk and IDLE" is an optional checkbox.
    1. Re-run your Python installer from `python.org` (e.g., `python-3.12.x-amd64.exe`).
    2. Select **Modify**.
    3. Ensure the checkbox for **tcl/tk and IDLE** is checked.
    4. Complete the installation.

### 3.4 Missing DLLs for OpenCV
*   **Symptom**: `ImportError: DLL load failed while importing cv2` under Windows N editions or completely fresh OS installs.
*   **Fix**: OpenCV relies on the Media Foundation pack. Install the "Media Feature Pack" via Windows Optional Features.
