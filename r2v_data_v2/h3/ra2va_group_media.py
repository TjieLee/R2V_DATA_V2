"""Loopback media serving without private credentials or directory listings."""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_launch import MEDIA_POLICY, PRIVATE_DIRECTORY


class PrivateMediaHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/_r2va_media_policy":
            self.send_response(204)
            self.send_header("X-R2VA-Media-Policy", MEDIA_POLICY)
            self.end_headers()
            return
        super().do_GET()

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if PRIVATE_DIRECTORY in path.parts or PRIVATE_DIRECTORY in path.resolve().parts or path.is_dir():
            self.send_error(404)
            return None
        return super().send_head()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    handler = functools.partial(PrivateMediaHandler, directory=args.directory)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
