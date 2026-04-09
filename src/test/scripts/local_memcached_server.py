#!/usr/bin/env python3

import argparse
import socketserver


class MemcachedTextHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            line = self.rfile.readline()
            if not line:
                return
            if not line.endswith(b"\r\n"):
                self.wfile.write(b"CLIENT_ERROR bad command line format\r\n")
                continue

            line = line[:-2]
            if not line:
                continue

            parts = line.split()
            command = parts[0].lower()

            if command == b"get":
                self._handle_get(parts[1:])
            elif command == b"set":
                self._handle_set(parts)
            elif command == b"incr":
                self._handle_incr(parts)
            elif command == b"delete":
                self._handle_delete(parts)
            elif command == b"quit":
                return
            else:
                self.wfile.write(b"ERROR\r\n")

    def _handle_get(self, keys):
        for key in keys:
            value = self.server.store.get(key)
            if value is None:
                continue
            header = b"VALUE " + key + b" 0 " + str(len(value)).encode() + b"\r\n"
            self.wfile.write(header)
            self.wfile.write(value)
            self.wfile.write(b"\r\n")
        self.wfile.write(b"END\r\n")

    def _handle_set(self, parts):
        if len(parts) < 5:
            self.wfile.write(b"CLIENT_ERROR bad command line format\r\n")
            return

        key = parts[1]
        try:
            byte_count = int(parts[4])
        except ValueError:
            self.wfile.write(b"CLIENT_ERROR bad command line format\r\n")
            return

        value = self.rfile.read(byte_count)
        terminator = self.rfile.read(2)
        if len(value) != byte_count or terminator != b"\r\n":
            self.wfile.write(b"CLIENT_ERROR bad data chunk\r\n")
            return

        self.server.store[key] = value
        if len(parts) < 6 or parts[5].lower() != b"noreply":
            self.wfile.write(b"STORED\r\n")

    def _handle_incr(self, parts):
        if len(parts) != 3:
            self.wfile.write(b"CLIENT_ERROR bad command line format\r\n")
            return

        key = parts[1]
        delta = int(parts[2])
        current = self.server.store.get(key)
        if current is None:
            self.wfile.write(b"NOT_FOUND\r\n")
            return

        try:
            next_value = int(current.decode()) + delta
        except ValueError:
            self.wfile.write(
                b"CLIENT_ERROR cannot increment or decrement non-numeric value\r\n"
            )
            return

        encoded = str(next_value).encode()
        self.server.store[key] = encoded
        self.wfile.write(encoded + b"\r\n")

    def _handle_delete(self, parts):
        if len(parts) != 2:
            self.wfile.write(b"CLIENT_ERROR bad command line format\r\n")
            return

        key = parts[1]
        if key in self.server.store:
            del self.server.store[key]
            self.wfile.write(b"DELETED\r\n")
        else:
            self.wfile.write(b"NOT_FOUND\r\n")


class ThreadedMemcachedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls):
        super().__init__(server_address, handler_cls)
        self.store = {
            b"serverNum": b"0",
            b"xmh-consistent-dsm": b"1",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    with ThreadedMemcachedServer((args.host, args.port), MemcachedTextHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
