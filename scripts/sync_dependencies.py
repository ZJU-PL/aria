#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script to synchronize dependencies between requirements.txt, setup.py, pyproject.toml, and meta.yaml.
This script reads dependencies from requirements.txt and updates the other files accordingly.
"""

import re
import yaml
import toml
import os


def parse_requirements(requirements_path):
    """Parse requirements.txt file and return a list of dependencies."""
    dependencies = []

    with open(requirements_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            dependencies.append(line)

    return dependencies


def update_setup_py(setup_path, dependencies):
    """Update dependencies in setup.py file."""
    with open(setup_path, "r") as f:
        content = f.read()

    # Find the REQUIRED list in setup.py
    required_pattern = r"REQUIRED\s*=\s*\[(.*?)\]"
    required_match = re.search(required_pattern, content, re.DOTALL)

    if required_match:
        # Format the dependencies as a multi-line list
        formatted_deps = []
        for dep in dependencies:
            # If the line already has quotes, use it as is, otherwise add quotes
            if "'" in dep or '"' in dep:
                formatted_deps.append(f"    {dep},")
            else:
                formatted_deps.append(f"    '{dep}',")

        new_required = "REQUIRED = [\n" + "\n".join(formatted_deps) + "\n]"

        # Replace the REQUIRED section in setup.py
        updated_content = content.replace(
            content[required_match.start() : required_match.end()], new_required
        )

        with open(setup_path, "w") as f:
            f.write(updated_content)

        print(f"Updated dependencies in {setup_path}")
    else:
        print(f"Could not find REQUIRED list in {setup_path}")


def update_pyproject_toml(pyproject_path, dependencies):
    """Update dependencies in pyproject.toml file."""
    try:
        # First read the file as text to preserve formatting
        with open(pyproject_path, "r") as f:
            content_text = f.read()

        # Also parse as TOML
        with open(pyproject_path, "r") as f:
            content = toml.load(f)

        # Format dependencies as they should appear in pyproject.toml
        formatted_deps = []
        for dep in dependencies:
            # Exclude any leading/trailing quotes if they exist
            dep = dep.strip("'").strip('"')
            formatted_deps.append(dep)

        # Update dependencies in pyproject.toml
        if "project" in content and "dependencies" in content["project"]:
            # Find the dependencies section in the original file
            pattern = r"(dependencies\s*=\s*\[).*?(\])"

            # Format the dependencies as multiple lines
            deps_text = "dependencies = [\n"
            for dep in formatted_deps:
                deps_text += f'    "{dep}",\n'
            deps_text += "]"

            # Replace the dependencies section in the original file
            updated_content = re.sub(pattern, deps_text, content_text, flags=re.DOTALL)

            with open(pyproject_path, "w") as f:
                f.write(updated_content)

            print(f"Updated dependencies in {pyproject_path}")
        else:
            print(f"Could not find dependencies section in {pyproject_path}")
    except Exception as e:
        print(f"Error updating {pyproject_path}: {str(e)}")


def update_meta_yaml(meta_path, dependencies):
    """Update dependencies in meta.yaml file."""
    with open(meta_path, "r") as f:
        content = f.read()

    # Parse the meta.yaml file
    # We need to handle Jinja2 templating, so yaml.load won't work directly
    # Instead, we'll update the requirements section using regex

    # Find the requirements:run section in meta.yaml
    run_pattern = r"(requirements:.*?run:.*?)(\s+-.*?)(\s+test:|\s*$)"
    run_match = re.search(run_pattern, content, re.DOTALL)

    if run_match:
        prefix = run_match.group(1)
        # Format dependencies as they should appear in meta.yaml
        formatted_deps = []
        for dep in dependencies:
            # Extract the package name and version specification
            # Check if the dependency contains version specifiers
            if any(spec in dep for spec in ["=", "<", ">", "~", "!"]):
                parts = re.match(r"([^=<>~!]+)(.+)", dep)
                if parts:
                    pkg_name = parts.group(1).strip()
                    version_spec = parts.group(2).strip()

                    # Handle different version specifications
                    if "~=" in version_spec:
                        version_spec = version_spec.replace("~=", ">=")

                    formatted_deps.append(f"    - {pkg_name} {version_spec}")
                else:
                    # Fallback in case regex fails
                    formatted_deps.append(f"    - {dep}")
            else:
                # If no version is specified, just use the dependency name as is
                formatted_deps.append(f"    - {dep}")

        # Create the new run section
        new_run_section = prefix + "\n" + "\n".join(formatted_deps)

        # Add the section that follows (test, about, etc.)
        if run_match.group(3):
            new_run_section += run_match.group(3)

        # Replace the run section in meta.yaml
        updated_content = content.replace(
            content[run_match.start() : run_match.end()], new_run_section
        )

        with open(meta_path, "w") as f:
            f.write(updated_content)

        print(f"Updated dependencies in {meta_path}")
    else:
        print(f"Could not find run section in {meta_path}")


def main():
    """Main function to synchronize dependencies."""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    setup_path = os.path.join(current_dir, "setup.py")
    pyproject_path = os.path.join(current_dir, "pyproject.toml")
    meta_path = os.path.join(current_dir, "meta.yaml")

    # Check if files exist
    if not os.path.exists(requirements_path):
        print(f"Error: {requirements_path} does not exist.")
        return

    # Parse requirements.txt
    dependencies = parse_requirements(requirements_path)

    # Update setup.py if it exists
    if os.path.exists(setup_path):
        update_setup_py(setup_path, dependencies)
    else:
        print(f"Warning: {setup_path} does not exist.")

    # Update pyproject.toml if it exists
    if os.path.exists(pyproject_path):
        update_pyproject_toml(pyproject_path, dependencies)
    else:
        print(f"Warning: {pyproject_path} does not exist.")

    # Update meta.yaml if it exists
    if os.path.exists(meta_path):
        update_meta_yaml(meta_path, dependencies)
    else:
        print(f"Warning: {meta_path} does not exist.")


if __name__ == "__main__":
    main()
