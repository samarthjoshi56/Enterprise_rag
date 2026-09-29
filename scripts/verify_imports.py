#!/usr/bin/env python3
"""Script to verify LangChain and LangGraph imports and print their versions."""

import sys


def main():
    print("=" * 60)
    print(" Verifying LangChain & LangGraph Framework Imports")
    print("=" * 60)

    all_passed = True

    modules_to_test = [
        ("langchain", "LangChain Core/Main Package"),
        ("langchain_core", "LangChain Core Abstractions"),
        ("langchain_community", "LangChain Community Integrations"),
        ("langgraph", "LangGraph Agentic Workflow Engine"),
    ]

    for mod_name, label in modules_to_test:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "N/A")
            print(f"  [OK] {label:<35} | Version: {ver}")
        except ImportError as e:
            print(f"  [FAILED] {label:<35} | Error: {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  ALL FRAMEWORK IMPORTS VERIFIED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("  SOME IMPORTS FAILED. Please check dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
