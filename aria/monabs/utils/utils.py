import os
from typing import Dict, List

def collect_smt2_files(root_dir: str) -> List[str]:
    smt2_files: List[str] = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".smt2"):
                smt2_files.append(os.path.join(root, name))
    return smt2_files