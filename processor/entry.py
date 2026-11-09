"""
Container entry point. Lambda-safe start-up: the only writable places are INPUT_DIR, OUTPUT_DIR,
HOME and TMPDIR (all set by the runtime), so the writable caches (matplotlib, numba, torch, XDG)
are pointed below $TMPDIR here, never at a fixed /tmp path, and USER is set for getpass. Then the
thread count is fixed from THREADS (or the cgroup CPU quota, never the host's core count) before
numpy or torch is imported, and main() runs once.
"""
import os
import sys
import tempfile

_tmp = tempfile.gettempdir()   # $TMPDIR when set
for _var, _sub in (("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "cache"),
                   ("TORCH_HOME", "torch")):
    if not os.environ.get(_var):
        os.environ[_var] = os.path.join(_tmp, _sub)
    os.makedirs(os.environ[_var], exist_ok=True)
_home = os.environ.get("HOME", "")
if not _home or not os.access(_home, os.W_OK):   # ECS: the image user has no writable home
    os.environ["HOME"] = os.path.join(_tmp, "home")
    os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ.setdefault("USER", "app")
os.environ.setdefault("LOGNAME", os.environ["USER"])

from processor.resources import resolve_threads, set_thread_env  # noqa: E402

_raw = os.environ.get("THREADS", "").strip()
_n, _source = resolve_threads(int(_raw) if _raw.isdigit() else 0)
set_thread_env(_n)

from processor.main import main  # noqa: E402

sys.exit(main())
