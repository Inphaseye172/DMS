#!/usr/bin/env python3
"""Verifies that every DLL in each DMS_deps backend folder can resolve its imports from the same
folder, the OS, the NVIDIA driver or the VC++ runtime (shipped with After Effects). Run after
scripts/fetch-deps.ps1:   python3 -I tools/check_deps.py build/DMS_deps
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe_imports import imports  # noqa: E402

OS_PREFIXES = ("api-ms-", "ext-ms-")
OS_DLLS = {
    "kernel32.dll", "advapi32.dll", "user32.dll", "gdi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll", "shlwapi.dll", "ws2_32.dll",
    "bcrypt.dll", "crypt32.dll", "version.dll", "rpcrt4.dll", "setupapi.dll", "dbghelp.dll", "psapi.dll", "ntdll.dll", "iphlpapi.dll",
    "secur32.dll", "wldap32.dll", "winmm.dll", "dxgi.dll", "d3d12.dll", "dxcore.dll", "cfgmgr32.dll", "comctl32.dll", "msvcrt.dll",
    "ucrtbase.dll", "nvcuda.dll",  # nvcuda.dll ships with the NVIDIA driver
}
VC_RUNTIME = ("vcruntime140", "msvcp140", "vcomp140", "concrt140")


def main(root):
    bad = 0
    for backend in ("ort-cuda", "ort-dml"):
        d = os.path.join(root, backend)
        if not os.path.isdir(d):
            print(f"{backend}: not present (skipped)")
            continue
        local = {f.lower() for f in os.listdir(d)}
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".dll"):
                continue
            with open(os.path.join(d, f), "rb") as fh:
                for kind, name in imports(fh.read()):
                    n = name.lower()
                    if n in local or n in OS_DLLS or n.startswith(OS_PREFIXES) or n.startswith(VC_RUNTIME):
                        continue
                    print(f"UNRESOLVED {backend}/{f}: {kind} import {name}")
                    bad += 1
        print(f"{backend}: {len([x for x in local if x.endswith('.dll')])} DLLs checked")
    print("OK" if bad == 0 else f"{bad} unresolved imports")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "DMS_deps")))
