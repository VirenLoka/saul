# fix_dns.py
import socket
_orig_getaddrinfo = socket.getaddrinfo

def _patched_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror:
        # If DNS fails, try treating host as a raw IP
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '', (host, port))]

socket.getaddrinfo = _patched_getaddrinfo