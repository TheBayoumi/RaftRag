"""
Deployment validation script.

Run this script to ensure the system meets all requirements before deployment.
"""

import subprocess
import sys
from pathlib import Path
from typing import List


class DeploymentValidator:
    """Validate deployment readiness."""

    def __init__(self) -> None:
        """Initialize validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_python_version(self) -> bool:
        """
        Validate Python version is 3.11+.

        Returns:
            bool: True if validation passes.
        """
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor >= 11:
            print(f"[OK] Python version: {sys.version}")
            return True

        self.errors.append(f"[ERROR] Python 3.11+ required, found {sys.version}")
        return False

    def validate_virtual_env(self) -> bool:
        """
        Validate running in virtual environment.

        Returns:
            bool: True if running in virtual environment.
        """
        if sys.prefix != sys.base_prefix:
            print(f"[OK] Running in virtual environment: {sys.prefix}")
            return True

        self.errors.append(
            "[ERROR] NOT running in virtual environment! This is MANDATORY!"
        )
        return False

    def validate_code_quality(self) -> bool:
        """
        Run code quality checks.

        Returns:
            bool: True if all checks pass.
        """
        checks = [
            ("black", ["black", "--check", "src/app/"]),
            ("isort", ["isort", "--check-only", "src/app/"]),
        ]

        all_passed = True
        for name, cmd in checks:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    print(f"[OK] {name} check passed")
                else:
                    self.warnings.append(f"[WARNING] {name} found issues")
                    all_passed = False
            except FileNotFoundError:
                self.warnings.append(f"[WARNING] {name} not found")
                all_passed = False

        return all_passed

    def validate_directories(self) -> bool:
        """
        Validate required directories exist.

        Returns:
            bool: True if all directories exist.
        """
        required_dirs = [
            "src/app",
            "data",
            "logs",
            "config",
            "tests",
        ]

        all_exist = True
        for directory in required_dirs:
            if Path(directory).exists():
                print(f"[OK] Directory exists: {directory}")
            else:
                self.errors.append(f"[ERROR] Missing directory: {directory}")
                all_exist = False

        return all_exist

    def validate_files(self) -> bool:
        """
        Validate required files exist.

        Returns:
            bool: True if all files exist.
        """
        required_files = [
            "pyproject.toml",
            ".gitignore",
            "requirements.txt",
            ".env.example",
            "src/app/main.py",
            "src/app/core/config.py",
        ]

        all_exist = True
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"[OK] File exists: {file_path}")
            else:
                self.errors.append(f"[ERROR] Missing file: {file_path}")
                all_exist = False

        return all_exist

    def run_validation(self) -> bool:
        """
        Run all validations.

        Returns:
            bool: True if all validations pass.
        """
        print("\n" + "=" * 60)
        print("DEPLOYMENT VALIDATION STARTING")
        print("=" * 60 + "\n")

        validations = [
            self.validate_python_version,
            self.validate_virtual_env,
            self.validate_directories,
            self.validate_files,
            self.validate_code_quality,
        ]

        all_passed = all(validation() for validation in validations)

        print("\n" + "=" * 60)
        if self.errors:
            print("VALIDATION FAILED - ERRORS:")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        if all_passed and not self.errors:
            print("[SUCCESS] ALL VALIDATIONS PASSED - READY FOR DEPLOYMENT")
        else:
            print("[FAILED] FIX ERRORS BEFORE DEPLOYMENT")

        print("=" * 60 + "\n")

        return all_passed and not self.errors


if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.run_validation()
    sys.exit(0 if success else 1)
