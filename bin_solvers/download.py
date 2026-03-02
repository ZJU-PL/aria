"""
Download pre-built SMT solvers and rename the binary solvers as 
cvc5, z3, and mathsat. 
TODO: support for sharpSAT (which may need to fetch the source code and build it)
"""
import os
import platform
import re
import shutil
import sys
import tarfile
import zipfile

import requests
from tqdm import tqdm

# yices2_mac_arm64 = "https://github.com/SRI-CSL/yices2/releases/download/Yices-2.6.4/yices-2.6.4-arm-apple-darwin20.6.0.tar.gz"
# yices2_mac = "https://github.com/SRI-CSL/yices2/releases/download/Yices-2.6.4/yices-2.6.4-x86_64-apple-darwin20.6.0.tar.gz"
# yices2_win64 = "https://github.com/SRI-CSL/yices2/releases/download/Yices-2.6.4/yices-2.6.4-x86_64-unknown-mingw32-static-gmp.zip"
# yices2_linux = "https://github.com/SRI-CSL/yices2/releases/download/Yices-2.6.4/yices-2.6.4-x86_64-pc-linux-gnu.tar.gz"


SOLVER_URLS = {
    'mac_arm64': {
        'cvc5': "https://github.com/cvc5/cvc5/releases/download/cvc5-1.0.3/cvc5-macOS-arm64",
        'z3': "https://github.com/Z3Prover/z3/releases/download/z3-4.10.2/z3-4.10.2-arm64-osx-11.0.zip",
        # 'mathsat': "https://mathsat.fbk.eu/download.php?file=mathsat-5.6.9-osx.tar.gz"
    },
    'mac_x64': {
        'cvc5': "https://github.com/cvc5/cvc5/releases/download/cvc5-1.0.3/cvc5-macOS",
        'z3': "https://github.com/Z3Prover/z3/releases/download/z3-4.10.2/z3-4.10.2-x64-osx-10.16.zip",
        # 'mathsat': "https://mathsat.fbk.eu/download.php?file=mathsat-5.6.9-osx.tar.gz"
    },
    'linux': {
        'cvc5': "https://github.com/cvc5/cvc5/releases/download/cvc5-1.0.3/cvc5-Linux",
        'z3': "https://github.com/Z3Prover/z3/releases/download/z3-4.10.2/z3-4.10.2-x64-glibc-2.31.zip",
        'mathsat': "https://mathsat.fbk.eu/release/mathsat-5.6.15-linux-x86_64.tar.gz"
    }
}


def get_binary_path(solver_name, archive_name):
    if solver_name == 'cvc5':
        return archive_name
    elif solver_name == 'z3':
        # Extract the full name without extension, handling decimal points properly
        match = re.search(r'(z3-\d+\.\d+\.\d+-[^.]+\.\d+)', archive_name)
        if match:
            base_name = match.group(1)
            return f'{base_name}/bin/z3'
    # elif solver_name == 'mathsat':
    #     match = re.search(r'mathsat-(\d+\.\d+\.\d+)-(.*?)(\.tar\.gz|\.zip)', archive_name)
    #     if match:
    #         version, platform, _ = match.groups()
    #        return f'mathsat-{version}-{platform}/bin/mathsat'
    return None


def get_os_type():
    system = platform.system().lower()
    if system == 'darwin':
        machine = platform.machine()
        if machine == 'arm64':
            return 'mac_arm64'
        return 'mac_x64'
    elif system == 'linux':
        return 'linux'
    return None


def download_file(url, output_file=None):
    if output_file is None:
        output_file = url.split('/')[-1]

    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as file, tqdm(
                desc=f"Downloading {os.path.basename(output_file)}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        return output_file
    except Exception as e:
        print(f"Failed to download: {url}")
        print(f"Error: {e}")
        return None


def extract_archive(filename):
    try:
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                total = len(zip_ref.filelist)
                with tqdm(total=total, desc=f"Extracting {filename}") as pbar:
                    for file in zip_ref.filelist:
                        zip_ref.extract(file)
                        pbar.update(1)
        elif filename.endswith('.tar.gz'):
            with tarfile.open(filename, 'r:gz') as tar_ref:
                total = len(tar_ref.getmembers())
                with tqdm(total=total, desc=f"Extracting {filename}") as pbar:
                    for member in tar_ref.getmembers():
                        tar_ref.extract(member)
                        pbar.update(1)
        return True
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")
        return False


def find_binary(solver_name, archive_name):
    if solver_name == 'cvc5':
        return archive_name

    if archive_name.endswith('.zip') or archive_name.endswith('.tar.gz'):
        binary_path = get_binary_path(solver_name, archive_name)
        print("find path: ", binary_path)
        if binary_path and os.path.exists(binary_path):
            return binary_path
    return None


def get_extracted_dir(solver_name, archive_name):
    # if solver_name == 'mathsat':
    #     match = re.search(r'mathsat-(\d+\.\d+\.\d+)-(.*?)(\.tar\.gz|\.zip)', archive_name)
    #     if match:
    #         version, platform, _ = match.groups()
    #         return f'mathsat-{version}-{platform}'
    if solver_name == 'z3':
        match = re.search(r'(z3-\d+\.\d+\.\d+-.*?)(\.zip|\.tar\.gz)', archive_name)
        if match:
            return match.group(1)
    return archive_name.replace('.zip', '').replace('.tar.gz', '')


def setup_solvers():
    DOWNLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
    os_type = get_os_type()
    if os_type not in SOLVER_URLS:
        print(f"Unsupported operating system: {platform.system()}")
        return False

    success = True
    total_steps = len(SOLVER_URLS[os_type])

    print(f"\nDetected OS: {os_type}")
    print(f"Setting up {total_steps} solvers...\n")

    for solver_name, url in SOLVER_URLS[os_type].items():
        target_path = os.path.join(DOWNLOAD_DIR, solver_name)
        # skip if the binary already exists
        if os.path.exists(target_path):
            print(f"\nSolver '{solver_name}' binary already exists in {DOWNLOAD_DIR}. Skipping download.")
            continue

        print(f"\n{'=' * 50}")
        print(f"Setting up {solver_name.upper()}:")
        print(f"{'=' * 50}")

        # Step 1: Download
        print(f"\nStep 1: Downloading {solver_name}")
        downloaded_file = download_file(url)
        if not downloaded_file:
            success = False
            continue

        # Step 2: Extract (if necessary)
        if downloaded_file.endswith('.zip') or downloaded_file.endswith('.tar.gz'):
            print(f"\nStep 2: Extracting {solver_name}")
            if not extract_archive(downloaded_file):
                success = False
                continue
        else:
            print(f"\nStep 2: Extraction not needed for {solver_name}")

        # Step 3: Setup binary
        print(f"\nStep 3: Setting up binary for {solver_name}")
        binary_path = find_binary(solver_name, downloaded_file)
        if binary_path and os.path.exists(binary_path):
            target_name = solver_name
            shutil.copy2(binary_path, target_name)
            if os.path.exists(target_name):
                os.chmod(target_name, 0o755)
                print(f"✓ Binary setup successful")
            else:
                print(f"✗ Failed to copy binary for {solver_name}")
                success = False

            # Step 4: Cleanup
            print(f"\nStep 4: Cleaning up temporary files")
            if downloaded_file.endswith(('.zip', '.tar.gz')):
                os.remove(downloaded_file)
                extracted_dir = get_extracted_dir(solver_name, downloaded_file)
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)
            else:
                os.remove(downloaded_file)
            print(f"✓ Cleanup successful")
        else:
            print(f"✗ Could not find binary for {solver_name}")
            success = False

        print(f"\nStatus: {'✓ Success' if success else '✗ Failed'}")

    # Make sure all binaries have proper permissions
    # for solver_name in ['cvc5', 'z3', 'mathsat']:
    for solver_name in ['cvc5', 'z3']:
        binary_path = os.path.join(DOWNLOAD_DIR, solver_name)
        if os.path.exists(binary_path):
            try:
                os.chmod(binary_path, 0o755)
            except Exception as e:
                print(f"Warning: Could not set permissions for {solver_name}: {e}")
                
    return success


if __name__ == '__main__':

    try:
        if setup_solvers():
            print("\n✓ All solvers have been successfully set up")
        else:
            print("\n✗ Some errors occurred during setup")
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        sys.exit(1)
