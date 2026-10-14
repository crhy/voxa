"""Forwards 127.0.0.1:<port> to the face server's Unix socket. Runs OUTSIDE the sandbox and contains no model
code: it only copies bytes, so the sandboxed neural code never needs network access.
Usage: python3 face_relay.py <unix-socket-path> [port=8765]"""
import socket
import sys
import threading


def pump(src, dst):
    try:
        while True:
            data = src.recv(1 << 16)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main():
    path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(4)
    print(f"relay 127.0.0.1:{port} -> {path}", flush=True)
    while True:
        client, _ = listener.accept()
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(path)
        except OSError:
            client.close()  # the face server is not up: the client sees "not available"
            continue
        threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
        threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


if __name__ == "__main__":
    main()
