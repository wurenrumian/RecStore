import os
import unittest
from unittest import mock

import ps_server_helpers


class TestPSServerHelpers(unittest.TestCase):
    def test_should_skip_server_start_raises_when_ports_partially_open(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                ps_server_helpers,
                'get_ports_from_config',
                return_value=[15000, 15001],
            ):
                with mock.patch.object(
                    ps_server_helpers,
                    'check_ps_server_running',
                    return_value=(True, [15000]),
                ):
                    with self.assertRaisesRegex(RuntimeError, 'partially available'):
                        ps_server_helpers.should_skip_server_start()


if __name__ == '__main__':
    unittest.main()
