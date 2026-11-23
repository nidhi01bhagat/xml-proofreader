"""Session-wide network guard for the offline test suite (S4.8c, decision D2).

Installed when this file is imported (before any test module), removed at ``pytest_unconfigure``:

* ``socket.socket.connect`` / ``connect_ex`` / ``sendto`` and ``socket.getaddrinfo`` refuse every non-loopback
  destination with :class:`NetworkBlockedError` (an ``OSError``, as a real network failure would be).
* Loopback (127.0.0.0/8, ``::1``, ``localhost``) and ``AF_UNIX`` stay allowed, so ``socket.socketpair()`` (which
  Windows implements over loopback) and local test servers keep working; subprocess pipes use no sockets.
* Every refused attempt is recorded. An autouse fixture FAILS the test that made it, even if the code under test
  swallowed the ``OSError`` (for example by mapping it to a passthrough) - a hidden network call cannot pass.
  Tests that provoke a refusal on purpose consume the record through the ``blocked_attempts`` fixture.

The guard covers this process only. Child processes do not inherit it; the subprocess tests that need network
isolation install their own guard in the child (tests/test_replay_fixture.py).
"""

from __future__ import annotations

import ipaddress
import socket

import pytest


class NetworkBlockedError(OSError):
    """A non-loopback network operation was attempted during the offline test suite."""


_ATTEMPTS: list[str] = []
_REAL = {
    "connect": socket.socket.connect,
    "connect_ex": socket.socket.connect_ex,
    "sendto": socket.socket.sendto,
    "getaddrinfo": socket.getaddrinfo,
}


def is_loopback_host(host: object) -> bool:
    if host is None:
        return True  # getaddrinfo(None, ...) resolves the local wildcard / loopback only
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    text = str(host).strip("[]").split("%", 1)[0]
    if text.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False  # any other host name would need DNS


def _refuse(operation: str, host: object) -> None:
    _ATTEMPTS.append(f"{operation} {host!r}")
    raise NetworkBlockedError(f"network access is blocked in the offline test suite ({operation} to {host!r})")


def _check(sock: socket.socket, address: object, operation: str) -> None:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not is_loopback_host(host):
        _refuse(operation, host)


def _connect(self, address):
    _check(self, address, "connect")
    return _REAL["connect"](self, address)


def _connect_ex(self, address):
    _check(self, address, "connect_ex")
    return _REAL["connect_ex"](self, address)


def _sendto(self, data, *args):
    if args:
        _check(self, args[-1], "sendto")
    return _REAL["sendto"](self, data, *args)


def _getaddrinfo(host, *args, **kwargs):
    if not is_loopback_host(host):
        _refuse("getaddrinfo", host)
    return _REAL["getaddrinfo"](host, *args, **kwargs)


def install() -> None:
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.socket.sendto = _sendto
    socket.getaddrinfo = _getaddrinfo


def uninstall() -> None:
    socket.socket.connect = _REAL["connect"]
    socket.socket.connect_ex = _REAL["connect_ex"]
    socket.socket.sendto = _REAL["sendto"]
    socket.getaddrinfo = _REAL["getaddrinfo"]


install()


def pytest_unconfigure(config) -> None:
    uninstall()


@pytest.fixture
def blocked_attempts() -> list[str]:
    """The live list of refused attempts; a test that provokes one on purpose clears it after its assertions."""
    return _ATTEMPTS


@pytest.fixture(autouse=True)
def _fail_on_network_attempts():
    start = len(_ATTEMPTS)
    yield
    leaked = _ATTEMPTS[start:]
    del _ATTEMPTS[start:]
    if leaked:
        pytest.fail(f"test attempted network access (blocked): {leaked}", pytrace=False)
