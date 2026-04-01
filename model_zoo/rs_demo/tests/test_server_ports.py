from __future__ import annotations

import socket
import unittest

from model_zoo.rs_demo.runtime.server import choose_available_ports


class TestChooseAvailablePorts(unittest.TestCase):
    def test_return_preferred_when_free(self) -> None:
        with socket.socket() as s0, socket.socket() as s1:
            s0.bind(("127.0.0.1", 0))
            s1.bind(("127.0.0.1", 0))
            p0 = s0.getsockname()[1]
            p1 = s1.getsockname()[1]

        got0, got1 = choose_available_ports("127.0.0.1", p0, p1)
        self.assertEqual((got0, got1), (p0, p1))

    def test_fallback_when_preferred_busy(self) -> None:
        with socket.socket() as s0, socket.socket() as s1:
            s0.bind(("127.0.0.1", 0))
            p0 = s0.getsockname()[1]
            s1.bind(("127.0.0.1", p0 + 1))
            p1 = s1.getsockname()[1]

            got0, got1 = choose_available_ports("127.0.0.1", p0, p1)
            self.assertNotEqual((got0, got1), (p0, p1))
            self.assertNotEqual(got0, got1)


if __name__ == "__main__":
    unittest.main()

