"""Keep Windows from sleeping while a process is alive (no power settings are changed).

  python scripts/keep_awake.py <pid>

Holds ES_SYSTEM_REQUIRED until <pid> exits, then releases it. Closing the lid or pressing the
power button still sleeps the machine.
"""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x102


def main():
    pid = int(sys.argv[1])
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(SYNCHRONIZE, False, pid)
    if not h:
        sys.exit(f"process {pid} not found")
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print(f"{time.ctime()}: keeping awake while {pid} runs", flush=True)
    try:
        while k32.WaitForSingleObject(h, 60_000) == WAIT_TIMEOUT:
            pass
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)
        k32.CloseHandle(h)
    print(f"{time.ctime()}: {pid} finished, sleep allowed again", flush=True)


if __name__ == "__main__":
    main()
