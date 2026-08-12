import os
import posixpath
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, unquote

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.fileutil import atomic_replace

_DAV = "{DAV:}"
DEFAULT_CONCURRENCY = 16
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_ROOT_DIR, ".env")


def get_concurrency():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
    return int(os.environ.get("NEXTCLOUD_CONCURRENCY", DEFAULT_CONCURRENCY))

_PROPFIND_BODY = (
    '<?xml version="1.0"?>'
    '<d:propfind xmlns:d="DAV:">'
    "<d:prop><d:resourcetype/><d:getcontentlength/></d:prop>"
    "</d:propfind>"
)


class NextcloudEntry:
    __slots__ = ("path", "name", "is_dir", "size")

    def __init__(self, path, is_dir, size):
        self.path = path
        self.name = posixpath.basename(path)
        self.is_dir = is_dir
        self.size = size

    def __repr__(self):
        return f"NextcloudEntry(path={self.path!r}, is_dir={self.is_dir}, size={self.size})"


class NextcloudClient:

    def __init__(self, base_url, username, password, timeout=30, pool_size=None):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.timeout = timeout
        self._files_root = f"{self.base_url}/remote.php/dav/files/{quote(username)}/"

        self.session = requests.Session()
        self.session.auth = (username, password)
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(502, 503, 504))
        adapter = HTTPAdapter(pool_maxsize=pool_size or get_concurrency(), max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, remote_path):
        remote_path = remote_path.strip("/")
        quoted = quote(remote_path, safe="/")
        return self._files_root + quoted

    def _relpath_from_href(self, href):
        href = unquote(href)
        marker = f"/remote.php/dav/files/{self.username}/"
        idx = href.find(marker)
        relpath = href[idx + len(marker):] if idx != -1 else href
        return relpath.strip("/")

    def list_dir(self, remote_path):
        response = self.session.request(
            "PROPFIND",
            self._url(remote_path),
            data=_PROPFIND_BODY,
            headers={"Depth": "1", "Content-Type": "application/xml"},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()

        root = remote_path.strip("/")
        entries = []
        xml_root = ET.fromstring(response.content)
        for resp in xml_root.findall(f"{_DAV}response"):
            href = resp.find(f"{_DAV}href")
            if href is None:
                continue
            relpath = self._relpath_from_href(href.text or "")
            if relpath == root:
                continue

            prop = resp.find(f"{_DAV}propstat/{_DAV}prop")
            if prop is None:
                continue
            resourcetype = prop.find(f"{_DAV}resourcetype")
            is_dir = resourcetype is not None and resourcetype.find(f"{_DAV}collection") is not None
            size_el = prop.find(f"{_DAV}getcontentlength")
            size = int(size_el.text) if size_el is not None and size_el.text else 0
            entries.append(NextcloudEntry(relpath, is_dir, size))
        return entries

    def walk(self, top):
        entries = self.list_dir(top)
        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        yield top.strip("/"), dirs, files
        for d in dirs:
            if ".git" in d.path:
                continue
            yield from self.walk(d.path)

    def walk_parallel(self, top, max_workers=None, on_progress=None):
        max_workers = max_workers or get_concurrency()
        results = []
        frontier = [top]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while frontier:
                futures = {pool.submit(self.list_dir, path): path for path in frontier}
                listings = {}
                for future in as_completed(futures):
                    listings[futures[future]] = future.result()
                    if on_progress:
                        on_progress(1)

                next_frontier = []
                for path in frontier:
                    entries = listings[path]
                    dirs = [e for e in entries if e.is_dir]
                    files = [e for e in entries if not e.is_dir]
                    results.append((path.strip("/"), dirs, files))
                    for d in dirs:
                        if ".git" not in d.path:
                            next_frontier.append(d.path)
                frontier = next_frontier
        return results

    def read_text(self, remote_path, encoding="utf-8"):
        response = self.session.get(self._url(remote_path), timeout=self.timeout)
        response.raise_for_status()
        return response.content.decode(encoding)

    def read_text_many(self, remote_paths, max_workers=None, on_progress=None):
        max_workers = max_workers or get_concurrency()
        remote_paths = list(remote_paths)
        if not remote_paths:
            return {}
        contents = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(remote_paths))) as pool:
            futures = {pool.submit(self.read_text, path): path for path in remote_paths}
            for future in as_completed(futures):
                contents[futures[future]] = future.result()
                if on_progress:
                    on_progress(1)
        return contents

    def download(self, remote_path, local_path):
        response = self.session.get(self._url(remote_path), timeout=self.timeout, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = local_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        atomic_replace(tmp_path, local_path)


_client = None


def get_nextcloud_client():
    global _client
    if _client is None:
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
        url = os.environ.get("NEXTCLOUD_URL")
        user = os.environ.get("NEXTCLOUD_USER")
        password = os.environ.get("NEXTCLOUD_PASS")
        if not url or not user or not password:
            raise RuntimeError(
                "NEXTCLOUD_URL, NEXTCLOUD_USER and NEXTCLOUD_PASS must be set (in .env or the environment) "
                "to use the Nextcloud data source."
            )
        _client = NextcloudClient(url, user, password)
    return _client
