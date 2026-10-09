import sys

from PyMemoryEditor.process.abstract import AbstractProcess


if sys.platform == "linux":

    def resolve_module_base(proc: AbstractProcess, proc_name: str) -> int | None:
        """Resolve the base address of `proc_name`'s main module."""
        for region in proc.get_memory_regions():
            # skip non-readable regions
            if b"r" not in region["struct"].Privileges:
                continue

            path: bytes = region["struct"].Path or b""
            if proc_name.encode() in path:
                return region["address"]

        return None


if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt

    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi

    _PROCESS_QUERY_INFORMATION: int = 0x0400
    _PROCESS_VM_READ: int = 0x0010
    _WIN_FALSE: wt.BOOL = wt.BOOL(0)

    def resolve_module_base(proc: AbstractProcess, proc_name: str) -> int | None:
        """Resolve the base address of `proc_name`'s main module."""
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, _WIN_FALSE, proc.pid)
        if not handle:
            return None

        try:
            name_buf = ctypes.create_unicode_buffer(512)

            modules = (wt.HMODULE * 1024)()
            needed = wt.DWORD()

            _psapi.EnumProcessModules(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed))
            count = needed.value // ctypes.sizeof(wt.HMODULE)

            for mod in modules[:count]:
                _psapi.GetModuleBaseNameW(handle, mod, name_buf, 260)
                if proc_name.lower() == name_buf.value.lower():
                    return mod
        finally:
            _kernel32.CloseHandle(handle)

        return None
