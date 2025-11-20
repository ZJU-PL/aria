#!/usr/bin/env python3
"""Script to build solvers from source code"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def build_sharpsat():
    """Download and build sharpSAT solver"""
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    build_dir = current_dir / "build"
    sharp_sat_dir = build_dir / "sharpSAT"
    
    # Create build directory if it doesn't exist
    build_dir.mkdir(exist_ok=True)
    
    # Clone sharpSAT repository
    if not sharp_sat_dir.exists():
        subprocess.run([
            "git", "clone", 
            "https://github.com/marcthurley/sharpSAT.git",
            str(sharp_sat_dir)
        ], check=True)
    
    # Build sharpSAT
    try:
        # Create build directory
        os.makedirs(sharp_sat_dir / "build", exist_ok=True)
        
        # Run cmake
        subprocess.run([
            "cmake", 
            "-DCMAKE_BUILD_TYPE=Release",
            ".."
        ], cwd=sharp_sat_dir / "build", check=True)
        
        # Run make
        subprocess.run([
            "make", "-j4"
        ], cwd=sharp_sat_dir / "build", check=True)
        
        # Copy binary to bin_solvers directory
        for sharp_sat_bin in ["sharpSAT", "sharp_sat"]:
            if os.path.exists(sharp_sat_dir / "build" / sharp_sat_bin):
                shutil.copy2(
                    sharp_sat_dir / "build" / sharp_sat_bin,
                    current_dir / "sharpSAT"
                )
                print(f"Successfully built sharpSAT: {sharp_sat_bin}")
                return True

        print("Failed to move sharpSAT binary to bin_solvers directory")
        return False

    except subprocess.CalledProcessError as e:
        print(f"Error building sharpSAT: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main entry point"""
    if not shutil.which("cmake"):
        print("Error: cmake is required but not found")
        sys.exit(1)
        
    if not shutil.which("make"):
        print("Error: make is required but not found")
        sys.exit(1)
    
    success = build_sharpsat()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
