"""S4.8c: the session-wide network guard (tests/conftest.py, decision D2) blocks external IPv4 and IPv6 traffic,
records every attempt, fails a test that hides one, and leaves loopback, socketpair and subprocesses working.

Only documentation addresses are used as "external" targets (192.0.2.0/24 TEST-NET-1, 2001:db8::/32), and every
attempt is refused before a packet is sent; no test here contacts a real service.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx2
import pytest

import conftest as guard

EXTERNAL_V4 = ("192.0.2.1", 443)
EXTERNAL_V6 = ("2001:db8::1", 443, 0, 0)


def consume(attempts: list[str]) -> list[str]:
    taken = list(attempts)
    attempts.clear()
    return taken


def test_guard_is_installed() -> None:
    assert socket.socket.connect is guard._connect and socket.getaddrinfo is guard._getaddrinfo


def test_external_ipv4_tcp_is_blocked(blocked_attempts) -> None:
    with pytest.raises(guard.NetworkBlockedError):
        socket.create_connection(EXTERNAL_V4, timeout=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(guard.NetworkBlockedError):
            s.connect(EXTERNAL_V4)
        with pytest.raises(guard.NetworkBlockedError):
            s.connect_ex(EXTERNAL_V4)
    assert consume(blocked_attempts) == ["getaddrinfo '192.0.2.1'", "connect '192.0.2.1'", "connect_ex '192.0.2.1'"]


def test_external_ipv6_tcp_is_blocked(blocked_attempts) -> None:
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("no IPv6 socket support on this machine")
    with s:
        with pytest.raises(guard.NetworkBlockedError):
            s.connect(EXTERNAL_V6)
    assert consume(blocked_attempts) == ["connect '2001:db8::1'"]


def test_external_udp_and_dns_are_blocked(blocked_attempts) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        with pytest.raises(guard.NetworkBlockedError):
            s.sendto(b"x", EXTERNAL_V4)
    with pytest.raises(guard.NetworkBlockedError):
        socket.getaddrinfo("api.anthropic.com", 443)
    assert consume(blocked_attempts) == ["sendto '192.0.2.1'", "getaddrinfo 'api.anthropic.com'"]


@pytest.mark.parametrize("url", ["http://192.0.2.1/", "http://[2001:db8::1]/"])
def test_the_http_stack_is_blocked_too(blocked_attempts, url) -> None:
    """httpx2 (the Anthropic SDK's HTTP library) with a real transport: refused before any packet."""
    with httpx2.Client(trust_env=False, timeout=1) as client:
        with pytest.raises(httpx2.HTTPError):
            client.get(url)
    assert consume(blocked_attempts), "the guard did not see the attempt"


def test_loopback_ipv4_tcp_still_works() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=5) as client:
            conn, _ = server.accept()
            with conn:
                client.sendall(b"ping")
                assert conn.recv(4) == b"ping"


def test_loopback_ipv6_tcp_still_works() -> None:
    try:
        server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        server.bind(("::1", 0))
    except OSError:
        pytest.skip("no IPv6 loopback on this machine")
    with server:
        server.listen(1)
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
            client.connect(server.getsockname()[:2])
            conn, _ = server.accept()
            with conn:
                client.sendall(b"pong")
                assert conn.recv(4) == b"pong"


def test_localhost_resolution_socketpair_and_subprocesses_still_work() -> None:
    assert socket.getaddrinfo("localhost", 0)
    a, b = socket.socketpair()  # loopback-based on Windows
    with a, b:
        a.sendall(b"ok")
        assert b.recv(2) == b"ok"
    proc = subprocess.run([sys.executable, "-I", "-c", "print('child ok')"], capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0 and proc.stdout.strip() == "child ok"


@pytest.mark.parametrize("host,loopback", [("127.0.0.1", True), ("127.5.6.7", True), ("::1", True), ("[::1]", True),
                                           ("localhost", True), (None, True), ("192.0.2.1", False),
                                           ("2001:db8::1", False), ("example.com", False), ("0.0.0.0", False)])
def test_loopback_classification(host, loopback) -> None:
    assert guard.is_loopback_host(host) is loopback


def test_a_swallowed_network_error_still_fails_the_test(tmp_path) -> None:
    """Positive control for the autouse check: a test that catches the OSError is still reported as failed."""
    (tmp_path / "conftest.py").write_text(Path(guard.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "test_hidden.py").write_text(textwrap.dedent("""
        import socket

        def test_hides_a_network_call():
            try:
                socket.create_connection(("192.0.2.1", 443), timeout=1)
            except OSError:
                pass  # e.g. mapped to a "connection" passthrough by the code under test

        def test_clean():
            assert True
    """), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path),
                           "-o", "addopts=", "--rootdir", str(tmp_path)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 1, proc.stdout  # the session fails
    # pytest reports the guard's teardown failure as an error of the hiding test (its call phase "passed")
    assert "1 error" in proc.stdout and "ERROR test_hidden.py::test_hides_a_network_call" in proc.stdout, proc.stdout
    assert "test_clean" not in proc.stdout.split("short test summary info")[-1], proc.stdout
    assert "attempted network access (blocked)" in proc.stdout
