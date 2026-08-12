import atexit
import json
import os
import threading
import time

from dotenv import load_dotenv

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_ROOT_DIR, ".env")
CACHE_DIR = os.path.join(_ROOT_DIR, ".cache", "nextcloud")

DEFAULT_MAX_SIZE_MB = 512


class NextcloudCache:

    def __init__(self, cache_dir=CACHE_DIR, max_size_bytes=None, delete_on_exit=None):
        self.cache_dir = cache_dir
        self.manifest_path = os.path.join(cache_dir, f"_manifest.{os.getpid()}.json")
        self.max_size_bytes = max_size_bytes if max_size_bytes is not None else _read_max_size_bytes()
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)
        self._manifest = self._load_manifest()
        self._total = sum(meta["size"] for meta in self._manifest.values())

        if delete_on_exit is None:
            delete_on_exit = _read_delete_on_exit()
        if delete_on_exit:
            atexit.register(self.clear)

    def _load_manifest(self):
        if not os.path.isfile(self.manifest_path):
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, retries=8, delay=0.05):
        tmp_path = self.manifest_path + ".tmp"
        for attempt in range(retries):
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self._manifest, f)
                os.replace(tmp_path, self.manifest_path)
                return
            except (PermissionError, FileNotFoundError):
                if attempt == retries - 1:
                    raise
                time.sleep(delay)

    def _local_path(self, relpath):
        return os.path.join(self.cache_dir, *relpath.split("/"))

    def _remove_entry(self, relpath):
        meta = self._manifest.pop(relpath, None)
        if meta is None:
            return
        self._total -= meta["size"]
        local_path = self._local_path(relpath)
        try:
            os.remove(local_path)
        except OSError:
            pass

    def _ensure_capacity(self, needed_size):
        if self._total + needed_size <= self.max_size_bytes:
            return
        for relpath, _ in sorted(self._manifest.items(), key=lambda kv: kv[1]["atime"]):
            if self._total + needed_size <= self.max_size_bytes:
                break
            self._remove_entry(relpath)
        if needed_size > self.max_size_bytes:
            print(
                f"[nextcloud_cache] warning: file of {needed_size} bytes alone exceeds the "
                f"{self.max_size_bytes} byte cache budget; keeping it cached anyway."
            )

    def get(self, client, remote_path, size_hint=None):
        relpath = remote_path.strip("/")
        local_path = self._local_path(relpath)

        with self._lock:
            meta = self._manifest.get(relpath)
            if meta is not None and os.path.isfile(local_path):
                meta["atime"] = time.time()
                return local_path
            if meta is None and os.path.isfile(local_path):
                size = os.path.getsize(local_path)
                self._manifest[relpath] = {"size": size, "atime": time.time()}
                self._total += size
                return local_path

        with self._lock:
            self._ensure_capacity(size_hint or 0)

        client.download(remote_path, local_path)
        actual_size = os.path.getsize(local_path)

        with self._lock:
            self._manifest[relpath] = {"size": actual_size, "atime": time.time()}
            self._total += actual_size
            self._save_manifest()

        return local_path

    def clear(self):
        with self._lock:
            for relpath in list(self._manifest.keys()):
                self._remove_entry(relpath)
            try:
                os.remove(self.manifest_path)
            except OSError:
                pass

    def current_size(self):
        return self._total


def _read_max_size_bytes():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
    mb = float(os.environ.get("NEXTCLOUD_CACHE_MAX_SIZE_MB", DEFAULT_MAX_SIZE_MB))
    return int(mb * 1024 * 1024)


def _read_delete_on_exit():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
    return os.environ.get("NEXTCLOUD_DELETE_CACHE_ON_EXIT", "false").strip().lower() in ("1", "true", "yes")


_cache = None


def get_nextcloud_cache():
    global _cache
    if _cache is None:
        _cache = NextcloudCache()
    return _cache
