#!/usr/bin/env python3
"""
Cross-platform Python virtual environment setup script.
"""

import os
import sys
import subprocess
import platform
import shutil
import re
from pathlib import Path

CUDA_VERSION = "12.8"
PYTHON_VERSION = "3.12"


def run_command(cmd, shell=False, check=True):
    """Run a command and handle errors gracefully."""
    try:
        if isinstance(cmd, str):
            print(f"Running: {cmd}")
        else:
            print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, shell=shell, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def find_cuda_installation(version="12.8"):
    """Find the best CUDA installation path, prioritizing the specified version."""
    cuda_installations = []

    # Method 1: Check current nvcc if available
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        # Extract CUDA version from nvcc output
        version_match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if version_match:
            cuda_version = version_match.group(1)
            # Try to find the installation path
            # Use 'which' on Linux/Mac, 'where' on Windows
            which_cmd = "where" if platform.system() == "Windows" else "which"
            nvcc_path = subprocess.run(
                [which_cmd, "nvcc"], capture_output=True, text=True, check=True
            ).stdout.strip().split('\n')[0]  # Take first result if multiple
            if nvcc_path:
                cuda_path = Path(
                    nvcc_path
                ).parent.parent  # nvcc is in bin/, so go up two levels
                cuda_installations.append((cuda_version, cuda_path))
                print(f"Found CUDA {cuda_version} via nvcc at {cuda_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Check common CUDA installation paths
    if platform.system() == "Windows":
        cuda_base_paths = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        ]
        # Also check CUDA_PATH if set
        if os.environ.get("CUDA_PATH"):
            cuda_base_paths.append(Path(os.environ.get("CUDA_PATH")))
    else:
        cuda_base_paths = [
            Path("/usr/local/cuda"),
            Path("/opt/cuda"),
            Path("/usr/lib/cuda"),
        ]

    for base_path in cuda_base_paths:
        if base_path and base_path.exists():
            # Look for version directories
            try:
                for item in base_path.iterdir():
                    if item.is_dir():
                        # Extract version from directory name (e.g., v12.8, cuda-12.8, 12.8)
                        version_match = re.search(r"(\d+\.\d+)", item.name)
                        if version_match:
                            version = version_match.group(1)
                            # Check if this looks like a valid CUDA installation (has bin/nvcc)
                            nvcc_path = (
                                item
                                / "bin"
                                / (
                                    "nvcc.exe"
                                    if platform.system() == "Windows"
                                    else "nvcc"
                                )
                            )
                            if nvcc_path.exists():
                                cuda_installations.append((version, item))
                                print(f"Found CUDA {version} at {item}")
            except PermissionError:
                pass

            # Also check if base_path itself is a CUDA installation
            nvcc_path = (
                base_path
                / "bin"
                / ("nvcc.exe" if platform.system() == "Windows" else "nvcc")
            )
            if nvcc_path.exists():
                version_match = re.search(r"(\d+\.\d+)", str(base_path))
                if version_match:
                    version = version_match.group(1)
                    cuda_installations.append((version, base_path))
                    print(f"Found CUDA {version} at {base_path}")

    if not cuda_installations:
        print("No CUDA installation found on system")
        return None, None

    # Remove duplicates and sort by version, prioritizing specified version
    unique_installations = list(
        dict(cuda_installations).items()
    )  # Remove duplicates by path

    def version_priority(item):
        ver, path = item
        major, minor = map(int, ver.split("."))
        # Prioritize specified version, then newer versions
        if ver == version:
            return (100, major, minor)  # High priority for specified version
        return (0, major, minor)  # Normal priority for others

    unique_installations.sort(key=version_priority, reverse=True)
    best_version, best_path = unique_installations[0]

    available_versions = [v for v, _ in unique_installations]
    print(f"Available CUDA versions: {', '.join(available_versions)}")
    if best_version == version:
        print(f"Selected preferred CUDA {best_version} at {best_path}")
    else:
        print(f"Selected CUDA {best_version} at {best_path}")

    return best_version, best_path


def setup_cuda_environment(cuda_path, cuda_version):
    """Set up CUDA environment variables for the current process and subprocess."""
    if not cuda_path or not cuda_version:
        return

    cuda_path = Path(cuda_path)

    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = str(cuda_path)
    os.environ["CUDA_HOME"] = str(cuda_path)  # Alternative name some tools use

    # Add CUDA bin to PATH if not already there
    cuda_bin = cuda_path / "bin"
    if cuda_bin.exists():
        current_path = os.environ.get("PATH", "")
        if str(cuda_bin) not in current_path:
            os.environ["PATH"] = f"{cuda_bin}{os.pathsep}{current_path}"
            print(f"Added {cuda_bin} to PATH")

    # Add CUDA lib to library path
    if platform.system() == "Windows":
        cuda_lib = cuda_path / "lib" / "x64"
        if cuda_lib.exists():
            current_path = os.environ.get("PATH", "")
            if str(cuda_lib) not in current_path:
                os.environ["PATH"] = f"{cuda_lib}{os.pathsep}{current_path}"
    else:
        cuda_lib = cuda_path / "lib64"
        if cuda_lib.exists():
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if str(cuda_lib) not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{cuda_lib}:{current_ld_path}"
                    if current_ld_path
                    else str(cuda_lib)
                )

    print(f"CUDA environment configured for version {cuda_version}")
    print(f"  CUDA_PATH = {os.environ.get('CUDA_PATH')}")
    print(f"  CUDA_HOME = {os.environ.get('CUDA_HOME')}")


def find_python(version="3.12"):
    """Find Python executable on the system."""

    version_no_dot = version.replace(".", "")
    possible_names = [
        f"python{version}",
        f"python{version_no_dot}",
        f"py -{version}",
        f"python{version}.exe",
        f"python{version_no_dot}.exe",
    ]

    # Check common installation paths on Windows
    if platform.system() == "Windows":
        common_paths = [
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Python"
            / f"Python{version_no_dot}"
            / "python.exe",
            Path(f"C:/Python{version_no_dot}/python.exe"),
            Path(f"C:/Program Files/Python{version_no_dot}/python.exe"),
            Path(f"C:/Program Files (x86)/Python{version_no_dot}/python.exe"),
        ]
    else:
        common_paths = [
            Path(f"/usr/bin/python{version}"),
            Path(f"/usr/bin/python{version_no_dot}"),
            Path(f"/usr/local/bin/python{version}"),
            Path(f"/usr/local/bin/python{version_no_dot}"),
        ]

    # Check if any of these paths exist
    for path in common_paths:
        if path.exists():
            # Verify it's actually Python of the correct version
            try:
                result = subprocess.run(
                    [str(path), "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if f"Python {version}" in result.stdout:
                    print(f"Found Python {version} at: {path}")
                    return str(path)
            except:
                continue

    # Try common command names
    for name in possible_names:
        try:
            if name.startswith("py "):
                # Handle py launcher specially
                result = subprocess.run(
                    name.split(), capture_output=True, text=True, check=True
                )
            else:
                result = subprocess.run(
                    [name, "--version"], capture_output=True, text=True, check=True
                )

            if f"Python {version}" in result.stdout:
                print(f"Found Python {version} using command: {name}")
                return name if not name.startswith("py ") else f"py -{version}"
        except:
            continue

    return None


def check_python_version(version="3.12"):
    major, minor = map(int, version.split("."))
    current_version = sys.version_info
    if current_version.major != major or current_version.minor != minor:
        print(
            f"Warning: Current Python {current_version.major}.{current_version.minor} detected."
        )
        print(f"This setup requires Python {version}.")

        # Try to find Python of the required version
        python_path = find_python(version)
        if python_path:
            print(f"Found Python {version}: {python_path}")
            return python_path
        else:
            print(f"Python {version} not found on system.")
            print(
                f"Please install Python {version} from https://www.python.org/downloads/"
            )
            response = input("Continue with current Python version anyway? (y/N): ")
            if response.lower() != "y":
                sys.exit(1)
            return sys.executable
    else:
        print(f"Python {major}.{minor} - Compatible!")
        return sys.executable


def create_venv(venv_path, python_executable):
    """Create virtual environment."""
    print(f"Creating virtual environment at: {venv_path}")
    print(f"Using Python executable: {python_executable}")

    # Create venv
    if python_executable.startswith("py "):
        # Handle py launcher
        cmd = python_executable.split() + ["-m", "venv", str(venv_path)]
    else:
        cmd = [python_executable, "-m", "venv", str(venv_path)]

    result = run_command(cmd)
    if result is None:
        print("Failed to create virtual environment")
        sys.exit(1)

    return venv_path


def get_venv_python(venv_path):
    """Get the path to Python executable in the virtual environment."""
    system = platform.system()
    if system == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def get_venv_pip(venv_path):
    """Get the path to pip executable in the virtual environment."""
    system = platform.system()
    if system == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def install_requirements(venv_path, cuda_path=None, cuda_version=None):
    """Install requirements in stages to handle PyTorch dependencies correctly."""
    pip_exe = get_venv_pip(venv_path)
    requirements_file = Path("requirements.txt")
    torch_requirements = Path("requirements_torch.txt")
    torch_build_requirements = Path("requirements_torch_build.txt")

    if not requirements_file.exists():
        print("requirements.txt not found!")
        print("Please ensure requirements.txt exists with all necessary dependencies.")
        sys.exit(1)

    # Set up CUDA environment before installing packages
    if cuda_path and cuda_version:
        setup_cuda_environment(cuda_path, cuda_version)

    print("Installing requirements in stages to handle PyTorch dependencies...")

    print("Stage 1: Installing PyTorch with CUDA support...")

    run_command(
        [str(pip_exe), "install", "-r", str(torch_requirements), "--no-cache-dir"]
    )

    print("Stage 2: Installing PyTorch-dependent packages...")

    run_command(
        [
            str(pip_exe),
            "install",
            "-r",
            str(torch_build_requirements),
            "--no-cache-dir",
            "--no-build-isolation",
        ]
    )

    print("Stage 3: Installing remaining packages...")

    run_command([str(pip_exe), "install", "-r", str(requirements_file)])

    print("All packages installed successfully!")


def main():
    print("Setting up Python virtual environment")
    print("=" * 70)

    # Detect CUDA installation
    print("Detecting CUDA installation...")
    cuda_version, cuda_path = find_cuda_installation(CUDA_VERSION)
    if cuda_version and cuda_path:
        print(f"Will use CUDA {cuda_version} at {cuda_path}")
    else:
        print("No CUDA detected, will use CPU-only PyTorch")
    print()

    # Check Python version and get the appropriate executable
    python_executable = check_python_version(PYTHON_VERSION)

    # Set up paths with machine-specific naming
    import socket

    hostname = socket.gethostname().lower()
    system = platform.system().lower()
    # Clean hostname to be filesystem-safe
    safe_hostname = "".join(c for c in hostname if c.isalnum() or c in "-_").rstrip()
    venv_name = f".venv_{safe_hostname}_{system}"
    venv_path = Path(venv_name)

    # Create virtual environment
    if venv_path.exists():
        response = input(
            f"Virtual environment '{venv_name}' already exists. Recreate? (y/N): "
        )
        if response.lower() == "y":
            shutil.rmtree(venv_path)
        else:
            print("Using existing virtual environment.")

    if not venv_path.exists():
        create_venv(venv_path, python_executable)

    # Install requirements with detected CUDA installation
    install_requirements(venv_path, cuda_path, cuda_version)

    print("\n" + "=" * 70)
    print("Setup completed successfully!")
    print("=" * 70)
    print(f"Virtual environment created at: {venv_path.absolute()}")
    if cuda_version and cuda_path:
        print(f"CUDA {cuda_version} configured at: {cuda_path}")
    else:
        print("PyTorch configured for CPU-only")

    print("\nTo activate the virtual environment:")
    system = platform.system()
    if system == "Windows":
        print(f"  PowerShell:      .\\{venv_path}\\Scripts\\Activate.ps1")
        print(f"  Command Prompt:  .\\{venv_path}\\Scripts\\activate.bat")
    else:
        print(f"  Bash/Shell:      source ./{venv_path}/bin/activate")


if __name__ == "__main__":
    main()
