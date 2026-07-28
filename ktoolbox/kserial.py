import typing
import re
import time

from typing import Optional
from typing import Union

from . import common

if typing.TYPE_CHECKING:
    from types import TracebackType
    import serial

logger = common.logger


class Serial:
    def __init__(
        self,
        port: Union[str, "serial.Serial"],
        baudrate: int = 115200,
        *,
        log_stream: Optional[typing.IO[bytes]] = None,
        own_log_stream: bool = False,
    ):
        try:
            import serial
        except ImportError as e:
            raise ImportError(
                "pyserial is required for the Serial class. "
                "Install with: pip install 'ktoolbox[pyserial]'"
            ) from e

        real_port: str
        real_ser: serial.Serial

        if isinstance(port, str):
            real_port = port
            real_ser = serial.Serial(port, baudrate=baudrate, timeout=0)
        else:
            # The caller can also pass in a serial.serial_for_url() instance
            # that they allocated themselves.
            real_ser = port
            real_ser.timeout = 0.0
            real_port = real_ser.port or "unspecified"

        self.port = real_port
        self._ser = real_ser
        self._bin_buf = b""
        self._str_buf: Optional[str] = None
        self._log_stream = log_stream
        self._own_log_stream = own_log_stream

    @property
    def buffer(self) -> str:
        if self._str_buf is None:
            self._str_buf = self._bin_buf.decode("utf-8", errors="surrogateescape")
        return self._str_buf

    @property
    def bin_buffer(self) -> bytes:
        return self._bin_buf

    def close(self) -> None:
        self._ser.close()
        if self._own_log_stream and self._log_stream is not None:
            self._log_stream.close()

    def send(
        self,
        msg: str,
        *,
        sleep: Optional[float] = None,
    ) -> None:
        logger.debug(f"serial[{self.port}]: send {repr(msg)}")
        self._ser.write(msg.encode("utf-8", errors="surrogateescape"))
        if sleep is not None:
            self.sleep(sleep)

    def read_all(self, *, max_read: Optional[int] = None) -> int:
        byte_readcount = 0
        while True:
            if max_read is not None:
                if byte_readcount >= max_read:
                    return byte_readcount
                readsize = min(100, max_read - byte_readcount)
            else:
                readsize = 100

            buf: bytes = self._ser.read(readsize)

            if buf:
                s = buf.decode("utf-8", errors="surrogateescape")
                logger.debug(
                    f"serial[{self.port}]: read buffer ({len(self._bin_buf)} + {len(buf)} bytes): {repr(s)}"
                )
                if self._log_stream is not None:
                    try:
                        self._log_stream.write(buf)
                    except Exception as e:
                        logger.warning(
                            f"serial[{self.port}]: failed to write to log stream: {e}"
                        )
                if not self._bin_buf:
                    self._str_buf = s
                elif (
                    self._str_buf
                    and not common.char_is_surrogateescaped(self._str_buf[-1])
                    and not common.char_is_surrogateescaped(s[0])
                ):
                    self._str_buf += s
                else:
                    self._str_buf = None
                self._bin_buf += buf
                byte_readcount += len(buf)

            if len(buf) < readsize:
                # Partial read. Return.
                #
                # The read data was appended to the internal self._bin_buf.
                return byte_readcount

    BACKLOG_SIZE_DEFAULT = 64 * 1024

    @typing.overload
    def expect(
        self,
        pattern: Union[str, re.Pattern[str]],
        timeout: Optional[float] = 30.0,
        *,
        verbose: bool = True,
        backlog_size: int = BACKLOG_SIZE_DEFAULT,
    ) -> str: ...

    @typing.overload
    def expect(
        self,
        pattern: None,
        timeout: Optional[float] = 30.0,
        *,
        verbose: bool = True,
        backlog_size: int = BACKLOG_SIZE_DEFAULT,
    ) -> None: ...

    @typing.overload
    def expect(
        self,
        pattern: Optional[Union[str, re.Pattern[str]]],
        timeout: Optional[float] = 30.0,
        *,
        verbose: bool = True,
        backlog_size: int = BACKLOG_SIZE_DEFAULT,
    ) -> Optional[str]: ...

    def expect(
        self,
        pattern: Optional[Union[str, re.Pattern[str]]],
        timeout: Optional[float] = 30.0,
        *,
        verbose: bool = True,
        backlog_size: int = BACKLOG_SIZE_DEFAULT,
    ) -> Optional[str]:
        import select

        start_timestamp = time.monotonic()

        # We use DOTALL like pexpect does.
        # If you need something else, compile the pattern yourself.
        #
        # See also https://pexpect.readthedocs.io/en/stable/overview.html#find-the-end-of-line-cr-lf-conventions
        pattern_re = common.as_regex(pattern, flags=re.DOTALL)

        if pattern_re is not None:
            logger.debug(
                f"serial[{self.port}]: expect message {repr(pattern)} (timeout {timeout})"
            )

        while True:
            self.read_all(max_read=4096)

            if pattern_re is not None:
                buffer = self.buffer
                match = re.search(pattern_re, buffer)
                if match:
                    end_idx = match.end()
                    consumed_chars = buffer[:end_idx]
                    consumed_bytes = consumed_chars.encode(
                        "utf-8",
                        errors="surrogateescape",
                    )
                    assert self._bin_buf.startswith(consumed_bytes)
                    logger.debug(
                        f"serial[{self.port}]: found expected message {len(consumed_bytes)} bytes, {len(self._bin_buf) - len(consumed_bytes)} bytes remaining (took {time.monotonic() - start_timestamp:.4f} seconds)"
                    )
                    self._str_buf = buffer[end_idx:]
                    self._bin_buf = self._bin_buf[len(consumed_bytes) :]
                    return consumed_chars

            if backlog_size > 0 and len(self._bin_buf) > 2 * backlog_size:
                consumed_len = len(self._bin_buf) - backlog_size
                self._str_buf = None
                self._bin_buf = self._bin_buf[consumed_len:]
                logger.debug(
                    f"serial[{self.port}]: drop excess {consumed_len} bytes, {len(self._bin_buf)} bytes remaining"
                )

            remaining_time = None
            if timeout is not None:
                remaining_time = (start_timestamp + timeout) - time.monotonic()

            if remaining_time is not None and remaining_time <= 0.0:
                if pattern_re is not None:
                    s = self._bin_buf.decode("utf-8", errors="surrogateescape")
                    if verbose:
                        logger.debug(
                            f"serial[{self.port}]: did not find expected message {repr(pattern)} after {time.monotonic() - start_timestamp:.4f} seconds (buffer content is {repr(s)})"
                        )
                    raise RuntimeError(
                        f"Did not receive expected message {repr(pattern)} within timeout (buffer content is {repr(s)})"
                    )
                return None

            _, _, _ = select.select([self._ser], [], [], remaining_time)

    def sleep(self, timeout: float) -> None:
        self.expect(None, timeout=timeout)

    def __enter__(self) -> "Serial":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional["TracebackType"],
    ) -> None:
        self.close()
