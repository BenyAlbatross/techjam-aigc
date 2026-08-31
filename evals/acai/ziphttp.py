"""Read a remote ZIP over HTTP range requests without downloading the whole archive.

ModelScope serves the WildFake archives from a CDN that supports byte ranges, so we
parse the central directory and pull only the members we actually need.
"""
import io
import re
import struct
import time
import urllib.request
import zlib

UA = {"User-Agent": "curl/8.5.0"}

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"


class RemoteZip:
    def __init__(self, url_resolver):
        """url_resolver() -> a fresh, directly-rangeable URL (CDN links expire)."""
        self._resolve = url_resolver
        self._url = url_resolver()
        self.size = self._head_size()

    def _open(self, headers):
        req = urllib.request.Request(self._url, headers={**UA, **headers})
        return urllib.request.urlopen(req, timeout=180)

    def _head_size(self):
        req = urllib.request.Request(self._url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(r.headers["Content-Length"])

    def get(self, start, length, retries=6):
        """Fetch [start, start+length) with retry + URL refresh on expiry."""
        end = start + length - 1
        for attempt in range(retries):
            try:
                with self._open({"Range": f"bytes={start}-{end}"}) as r:
                    if r.status != 206:
                        raise IOError(f"expected 206, got {r.status}")
                    buf = r.read()
                if len(buf) != length:
                    raise IOError(f"short read {len(buf)} != {length}")
                return buf
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
                self._url = self._resolve()

    def central_directory(self):
        """Return (cd_offset, cd_size, entry_count), handling Zip64."""
        tail_len = min(65_557, self.size)
        tail = self.get(self.size - tail_len, tail_len)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise ValueError("no EOCD found")
        count, cd_size, cd_off = struct.unpack("<H", tail[i + 10:i + 12])[0], \
            struct.unpack("<I", tail[i + 12:i + 16])[0], \
            struct.unpack("<I", tail[i + 16:i + 20])[0]

        if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
            j = tail.rfind(EOCD64_LOC_SIG, 0, i)
            if j < 0:
                raise ValueError("zip64 locator missing")
            eocd64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
            rec = self.get(eocd64_off, 56)
            if rec[:4] != EOCD64_SIG:
                raise ValueError("bad zip64 EOCD signature")
            count = struct.unpack("<Q", rec[32:40])[0]
            cd_size = struct.unpack("<Q", rec[40:48])[0]
            cd_off = struct.unpack("<Q", rec[48:56])[0]
        return cd_off, cd_size, count

    def entries(self, chunk=32 << 20, progress=None):
        """Yield dicts for every central-directory entry."""
        cd_off, cd_size, count = self.central_directory()
        blob = bytearray()
        got = 0
        while got < cd_size:
            n = min(chunk, cd_size - got)
            blob += self.get(cd_off + got, n)
            got += n
            if progress:
                progress(got, cd_size)
        return list(_parse_cd(bytes(blob), count))


def _parse_cd(blob, count):
    p = 0
    for _ in range(count):
        if blob[p:p + 4] != CEN_SIG:
            raise ValueError(f"bad central header at {p}")
        method = struct.unpack("<H", blob[p + 10:p + 12])[0]
        crc = struct.unpack("<I", blob[p + 16:p + 20])[0]
        csize = struct.unpack("<I", blob[p + 20:p + 24])[0]
        usize = struct.unpack("<I", blob[p + 24:p + 28])[0]
        nlen = struct.unpack("<H", blob[p + 28:p + 30])[0]
        elen = struct.unpack("<H", blob[p + 30:p + 32])[0]
        clen = struct.unpack("<H", blob[p + 32:p + 34])[0]
        lho = struct.unpack("<I", blob[p + 42:p + 46])[0]
        name = blob[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = blob[p + 46 + nlen:p + 46 + nlen + elen]

        # Zip64 extra field patches whichever fields are saturated, in fixed order.
        if 0xFFFFFFFF in (csize, usize, lho):
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    vals = extra[q + 4:q + 4 + hsz]
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                    break
                q += 4 + hsz
        yield {"name": name, "method": method, "csize": csize, "usize": usize,
               "crc": crc, "lho": lho}
        p += 46 + nlen + elen + clen


def modelscope_resolver(dataset, path, revision="master"):
    api = ("https://modelscope.cn/api/v1/datasets/"
           f"{dataset}/repo?Revision={revision}&FilePath={path}")

    def resolve():
        req = urllib.request.Request(api, headers=UA, method="HEAD")
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        op = urllib.request.build_opener(NoRedirect)
        try:
            with op.open(req, timeout=120) as r:
                loc = r.headers.get("Location")
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location")
        if not loc:
            raise IOError("no CDN redirect returned")
        return loc
    return resolve
