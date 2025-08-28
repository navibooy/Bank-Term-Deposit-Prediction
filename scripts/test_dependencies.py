#!/usr/bin/env python3
"""
Dependency Testing Script for Airflow Container

This script tests that all required ML dependencies are available in the Airflow container.
"""

import importlib
import sys
from typing import List, Tuple


def test_dependencies() -> Tuple[List[str], List[str]]:
    """Test that all required dependencies can be imported."""

    # List of required dependencies for the ML pipeline
    dependencies = [
        "catboost",
        "evidently",
        "fastapi",
        "kaggle",
        "matplotlib",
        "mlflow",
        "numpy",
        "pandas",
        "sklearn",  # scikit-learn imports as sklearn
        "seaborn",
        "shap",
        "yaml",  # pyyaml imports as yaml
        "airflow",
        "psycopg2",
    ]

    successful_imports = []
    failed_imports = []

    print("Testing dependency imports...")
    print("=" * 50)

    for dependency in dependencies:
        try:
            module = importlib.import_module(dependency)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {dependency}: {version}")
            successful_imports.append(f"{dependency}=={version}")
        except ImportError as e:
            print(f"❌ {dependency}: {str(e)}")
            failed_imports.append(dependency)
        except Exception as e:
            print(f"⚠️  {dependency}: {str(e)}")
            failed_imports.append(dependency)

    return successful_imports, failed_imports


def test_src_imports():
    """Test that project src modules can be imported."""

    print("\nTesting project module imports...")
    print("=" * 50)

    src_modules = [
        "src.data.ingest",
        "src.features.transform",
        "src.models.train",
        "src.models.validate",
    ]

    successful = []
    failed = []

    for module_name in src_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
            successful.append(module_name)
        except ImportError as e:
            print(f"❌ {module_name}: {str(e)}")
            failed.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}: {str(e)}")
            failed.append(module_name)

    return successful, failed


def main():
    """Main test function."""
    print("Dependency Test Report")
    print("=" * 50)

    # Test external dependencies
    successful_deps, failed_deps = test_dependencies()

    # Test project modules
    successful_modules, failed_modules = test_src_imports()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully imported dependencies: {len(successful_deps)}")
    print(f"❌ Failed dependency imports: {len(failed_deps)}")
    print(f"✅ Successfully imported project modules: {len(successful_modules)}")
    print(f"❌ Failed project module imports: {len(failed_modules)}")

    if failed_deps:
        print(f"\nFailed dependencies: {', '.join(failed_deps)}")

    if failed_modules:
        print(f"Failed modules: {', '.join(failed_modules)}")

    # Return appropriate exit code
    if failed_deps or failed_modules:
        print("\n🚨 Some imports failed. Check container dependencies.")
        sys.exit(1)
    else:
        print("\n🎉 All dependencies are available!")
        sys.exit(0)


if __name__ == "__main__":
    main()
