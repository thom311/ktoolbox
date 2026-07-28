import pytest
import socket

from ktoolbox import kserial


def test_serial_class() -> None:
    try:
        import serial
    except ModuleNotFoundError:
        pytest.skip("pyserial module not available")

    with pytest.raises(serial.serialutil.SerialException):
        kserial.Serial("")

    with socket.socket() as server_socket:
        server_socket.bind(("localhost", 0))
        server_socket.listen(1)
        port = server_socket.getsockname()[1]

        serial_port = serial.serial_for_url(f"socket://localhost:{port}")

        client_socket, _ = server_socket.accept()

        with (
            client_socket,
            kserial.Serial(serial_port) as ser,
        ):
            ser.send("hello", sleep=0)

            assert client_socket.recv(1024) == b"hello"

            client_socket.sendall("message_1".encode())

            m = ser.expect("_")
            assert m == "message_"

            m0 = ser.expect(None, timeout=0.0)
            assert m0 is None

            m = ser.expect(".+", timeout=0.0)
            assert m == "1"

            with pytest.raises(RuntimeError):
                ser.expect(".+", timeout=0.01)
