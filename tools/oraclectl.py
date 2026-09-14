#!/usr/bin/env python3
"""Send one request to the world's oracle socket and print the reply.

    sudo python3 tools/oraclectl.py '{"command": "prop_geometries"}' [/tmp/isaac-oracle.sock]

Local only: the socket is owner-only on the host. The evaluation harness
runs this over ssh (eval/host.py); nothing an agent holds reaches it."""

import json
import socket
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    req = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/isaac-oracle.sock"
    json.loads(req)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(60.0)
        s.connect(path)
        s.sendall((req.strip() + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    sys.stdout.write(buf.decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
