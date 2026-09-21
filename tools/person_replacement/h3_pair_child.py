"""Exec wrapper for Linux parent-death cleanup without unsafe preexec_fn."""

import ctypes
import os
import signal
import sys


def arm_parent_death(parent):
    if sys.platform == "linux":
        if ctypes.CDLL(None).prctl(1,signal.SIGTERM) != 0:
            raise OSError("PR_SET_PDEATHSIG failed")
        if os.getppid() != parent:
            raise SystemExit("Parent exited before child startup")


def main():
    arm_parent_death(int(sys.argv[1]))
    os.environ["R2V_PAIR_LAUNCHER_PID"] = str(os.getpid())
    os.execvpe(sys.argv[2],sys.argv[2:],os.environ)


if __name__ == "__main__":
    main()
