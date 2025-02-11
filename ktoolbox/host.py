import dataclasses
import logging
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import threading
import time
import typing

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import AnyStr
from typing import Callable
from typing import Optional
from typing import Union

from . import common

from .common import FutureThread


INTERNAL_ERROR_PREFIX = "Host.run(): "

RETURNCODE_INTERNAL = 1024
RETURNCODE_CANCELLED = 1025

TERMINATE_KILL_WAIT = 1.0

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_unique_log_id_value = 0


def _normalize_cmd(
    cmd: Union[str, Iterable[str]],
) -> Union[str, tuple[str, ...]]:
    if isinstance(cmd, str):
        return cmd
    else:
        return tuple(cmd)


def _normalize_env(
    env: Optional[Mapping[str, Optional[str]]],
) -> Optional[dict[str, Optional[str]]]:
    if env is None:
        return None
    return dict(env)


def _cmd_to_logstr(cmd: Union[str, tuple[str, ...]]) -> str:
    return repr(_cmd_to_shell(cmd))


def _cmd_to_shell(
    cmd: Union[str, Iterable[str]],
    *,
    cwd: Optional[str] = None,
) -> str:
    if not isinstance(cmd, str):
        cmd = shlex.join(cmd)
    if cwd is not None:
        cmd = f"cd {shlex.quote(cwd)} || exit 1 ; {cmd}"
    return cmd


def _cmd_to_argv(cmd: Union[str, Iterable[str]]) -> tuple[str, ...]:
    if isinstance(cmd, str):
        return ("/bin/sh", "-c", cmd)
    return tuple(cmd)


def _cmd_with_sudo(
    *,
    cmd: Union[str, Iterable[str]],
    env: Optional[Mapping[str, Optional[str]]],
    cwd: Optional[str],
) -> tuple[str, ...]:
    cmd2 = ["sudo", "-n"]

    if env:
        for k, v in env.items():
            assert k == shlex.quote(k)
            assert "=" not in k
            if v is not None:
                cmd2.append(f"{k}={v}")

    if cwd is not None:
        # sudo's "--chdir" option often does not work based on the sudo
        # configuration. Instead, change the directory inside the shell
        # script.
        cmd = _cmd_to_shell(cmd, cwd=cwd)

    cmd2.extend(_cmd_to_argv(cmd))

    return tuple(cmd2)


def _path_make_absolute(
    path: Union[str, os.PathLike[Any]],
    *,
    cwd: Optional[str] = None,
) -> str:
    path = str(path)
    if os.path.isabs(path):
        return path
    if cwd is None:
        # We cannot make this path absolute. The Host will determine
        # the base of the path, and it is rather unpredictable. Avoid
        # relative paths due to that.
        return path
    return os.path.join(cwd, path)


def _pr_kill(
    pr: subprocess.Popen[bytes],
    *,
    sudo: bool,
    handle_log: Callable[[int, str], None],
    with_sigkill: bool = False,
) -> None:
    try:
        if sudo:
            # If the process runs with `sudo`, we must also use `sudo` to send the
            # signal.  Also, if the signal originates from the same process
            # group, it is ignored (hence the "setsid").
            #
            # Also, note that `sudo` cannot intercept SIGKILL and cannot
            # forward it to the running process. So killing with SIGKILL is
            # unlikely to work reasonably.  In any case, SIGKILL is only a last
            # resort. The user is supposed to only cancel processes if they let
            # themselves be terminated by SIGTERM.
            subprocess.run(
                [
                    "setsid",
                    "sudo",
                    "-n",
                    "kill",
                    "-SIGKILL" if with_sigkill else "-SIGTERM",
                    str(pr.pid),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            if with_sigkill:
                pr.kill()
            else:
                pr.terminate()
    except Exception as e:
        handle_log(
            logging.ERROR,
            f"cancelled: sending {'SIGKILL' if with_sigkill else 'SIGTERM'} to process {pr.pid} failed: {e}",
        )


def _unique_log_id() -> int:
    # For each run() call, we log a message when starting the command and when
    # completing it. Add a unique number to those logging statements, so that
    # we can easier find them in a large log.
    with _lock:
        global _unique_log_id_value
        _unique_log_id_value += 1
        return _unique_log_id_value


@dataclass(frozen=True)
class _BaseResult(ABC, typing.Generic[AnyStr]):
    # _BaseResult only exists to have the first 3 parameters positional
    # arguments and the subsequent parameters (in BaseResult) marked as
    # common.KW_ONLY_DATACLASS. Once we no longer support Python 3.9, the
    # classes can be merged.
    out: AnyStr
    err: AnyStr
    returncode: int


@dataclass(frozen=True, **common.KW_ONLY_DATACLASS)
class BaseResult(_BaseResult[AnyStr]):
    # In most cases, "success" is the same as checking for returncode zero.  In
    # some cases, it can be overwritten to be of a certain value.
    forced_success: Optional[bool] = dataclasses.field(
        default=None,
        # kw_only=True <- use once we upgrade to 3.10 and drop KW_ONLY_DATACLASS.
    )

    cancelled: bool = False

    @property
    def success(self) -> bool:
        if self.forced_success is not None:
            return self.forced_success
        if self.cancelled:
            return False
        return self.returncode == 0

    def __bool__(self) -> bool:
        return self.success

    def debug_str(self, *, with_output: bool = True) -> str:
        if self.forced_success is None or self.forced_success == (self.returncode == 0):
            if self.success:
                status = "success"
            else:
                status = f"failed (exit {self.returncode})"
        else:
            if self.forced_success:
                status = f"success [forced] (exit {self.returncode})"
            else:
                status = "failed [forced] (exit 0)"

        out = ""
        if self.out and with_output:
            out = f"; out={repr(self.out)}"

        err = ""
        if self.err and with_output:
            err = f"; err={repr(self.err)}"

        return f"{status}{out}{err}"

    def debug_msg(self) -> str:
        return f"cmd {self.debug_str()}"

    def match(
        self,
        *,
        out: Optional[Union[AnyStr, re.Pattern[AnyStr]]] = None,
        err: Optional[Union[AnyStr, re.Pattern[AnyStr]]] = None,
        returncode: Optional[int] = None,
    ) -> bool:
        if returncode is not None:
            if self.returncode != returncode:
                return False

        def _check(
            val: AnyStr,
            compare: Optional[Union[AnyStr, re.Pattern[AnyStr]]],
        ) -> bool:
            if compare is None:
                return True
            if isinstance(compare, re.Pattern):
                return bool(compare.search(val))
            return val == compare

        return _check(self.out, out) and _check(self.err, err)


@dataclass(frozen=True)
class Result(BaseResult[str]):
    def dup_with_forced_success(self, forced_success: bool) -> "Result":
        if forced_success == self.success:
            return self
        return Result(
            self.out,
            self.err,
            self.returncode,
            forced_success=forced_success,
            cancelled=self.cancelled,
        )

    CANCELLED: typing.ClassVar["Result"]

    def _maybe_intern(self) -> "Result":
        if self.cancelled and self == Result.CANCELLED:
            assert self is not Result.CANCELLED
            return self.CANCELLED
        return self


@dataclass(frozen=True)
class BinResult(BaseResult[bytes]):
    def decode(self, errors: str = "strict") -> Result:
        return Result(
            self.out.decode(errors=errors),
            self.err.decode(errors=errors),
            self.returncode,
            forced_success=self.forced_success,
            cancelled=self.cancelled,
        )

    def dup_with_forced_success(self, forced_success: bool) -> "BinResult":
        if forced_success == self.success:
            return self
        return BinResult(
            self.out,
            self.err,
            self.returncode,
            forced_success=forced_success,
            cancelled=self.cancelled,
        )

    @staticmethod
    def new_internal(
        msg: str,
        returncode: int = RETURNCODE_INTERNAL,
        cancelled: bool = False,
    ) -> "BinResult":
        return BinResult(
            b"",
            (INTERNAL_ERROR_PREFIX + msg).encode(errors="surrogateescape"),
            returncode,
            cancelled=cancelled,
        )

    CANCELLED: typing.ClassVar["BinResult"]

    def _maybe_intern(self) -> "BinResult":
        if self.cancelled and self == BinResult.CANCELLED:
            return self.CANCELLED
        return self


BinResult.CANCELLED = BinResult.new_internal(
    "operation cancelled",
    RETURNCODE_CANCELLED,
    cancelled=True,
)

Result.CANCELLED = BinResult.CANCELLED.decode()


class Host(ABC):
    def __init__(
        self,
        *,
        sudo: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._has_sudo: Optional[bool] = None
        self.sudo = sudo

    @abstractmethod
    def pretty_str(self) -> str:
        pass

    @abstractmethod
    def _prepare_run(
        self,
        *,
        cmd: Union[str, Iterable[str]],
        env: Optional[Mapping[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        has_cancellable: bool,
    ) -> tuple[
        Union[str, tuple[str, ...]],
        Optional[dict[str, Optional[str]]],
        Optional[str],
    ]:
        pass

    @typing.overload
    def run(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: typing.Literal[True] = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[Callable[[Result], bool]] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
    ) -> Result: ...

    @typing.overload
    def run(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: typing.Literal[False],
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[Callable[[BinResult], bool]] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
    ) -> BinResult: ...

    @typing.overload
    def run(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: bool = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[
            Union[Callable[[Result], bool], Callable[[BinResult], bool]]
        ] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
    ) -> Union[Result, BinResult]: ...

    def run(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: bool = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[
            Union[Callable[[Result], bool], Callable[[BinResult], bool]]
        ] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
    ) -> Union[Result, BinResult]:
        log_id = _unique_log_id()

        if log_level is None:
            if log_level_result is None:
                log_level = logging.DEBUG
            else:
                log_level = -1

        sudo = self.get_effective_sudo(sudo)

        real_cmd, real_env, real_cwd = self._prepare_run(
            cmd=cmd,
            env=env,
            cwd=cwd,
            sudo=sudo,
            has_cancellable=cancellable is not None,
        )

        if log_level >= 0:
            logger.log(
                log_level,
                f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: call {_cmd_to_logstr(real_cmd)}",
            )

        if isinstance(log_lineoutput, bool):
            if not log_lineoutput:
                log_lineoutput = -1
            elif log_level >= 0:
                log_lineoutput = log_level
            else:
                log_lineoutput = logging.DEBUG

        if common.Cancellable.is_cancelled(cancellable):
            bin_result = BinResult.CANCELLED
        else:
            _handle_line = line_callback
            if log_lineoutput >= 0:
                line_num = [0, 0]

                def _handle_line_log(is_stdout: bool, line: bytes) -> None:
                    outtype = "stdout" if is_stdout else "stderr"
                    logger.log(
                        log_lineoutput,
                        f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: {outtype}[{line_num[is_stdout]}] {repr(line)}",
                    )
                    line_num[is_stdout] += 1
                    if line_callback is not None:
                        line_callback(is_stdout, line)

                _handle_line = _handle_line_log

            def _handle_log(level: int, msg: str) -> None:
                if level >= log_level:
                    logger.log(
                        level,
                        f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: {msg}",
                    )

            bin_result = self._run(
                cmd=real_cmd,
                env=real_env,
                cwd=real_cwd,
                sudo=sudo,
                handle_line=_handle_line,
                handle_log=_handle_log,
                cancellable=cancellable,
            )

        # The remainder is only concerned with printing a nice logging message and
        # (potentially) decode the binary output.

        str_result: Optional[Result] = None
        unexpected_binary = False
        is_binary = True
        decode_exception: Optional[Exception] = None
        if text:
            # The caller requested string (Result) output. "decode_errors" control what we do.
            #
            # - None (the default). We effectively use "errors='replace'"). On any encoding
            #   error we log an ERROR message.
            # - otherwise, we use "decode_errors" as requested. An encoding error will not
            #   raise the log level, but we will always log the result. We will even log
            #   the result if the decoding results in an exception (see decode_exception).
            try:
                # We first always try to decode strictly to find out whether
                # it's valid utf-8.
                str_result = bin_result.decode(errors="strict")
            except UnicodeError as e:
                if decode_errors == "strict":
                    decode_exception = e
                is_binary = True
            else:
                is_binary = False

            if decode_exception is not None:
                # We had an error. We keep this and re-raise later.
                pass
            elif not is_binary and (
                decode_errors is None
                or decode_errors in ("strict", "ignore", "replace", "surrogateescape")
            ):
                # We are good. The output is not binary, and the caller did not
                # request some unusual decoding. We already did the decoding.
                pass
            elif decode_errors is not None:
                # Decode again, this time with the decoding option requested
                # by the caller.
                try:
                    str_result = bin_result.decode(errors=decode_errors)
                except UnicodeError as e:
                    decode_exception = e
            else:
                # We have a binary and the caller didn't specify a special
                # encoding. We use "replace" fallback, but set a flag that
                # we have unexpected_binary (and lot an ERROR below).
                str_result = bin_result.decode(errors="replace")
                unexpected_binary = True

        if check_success is None:
            result_success = bin_result.success
        else:
            result_success = True
            if text:
                str_check = typing.cast(Callable[[Result], bool], check_success)
                if str_result is None:
                    # This can only happen in text mode when the caller specified
                    # a "decode_errors" that resulted in a "decode_exception".
                    # The function will raise an exception, and we won't call
                    # the check_success() handler.
                    #
                    # Avoid this by using text=False or a "decode_errors" value
                    # that does not fail.
                    result_success = False
                elif not str_check(str_result):
                    result_success = False
            else:
                bin_check = typing.cast(Callable[[BinResult], bool], check_success)
                if not bin_check(bin_result):
                    result_success = False

        status_msg = ""
        if log_level_fail is not None and not result_success:
            result_log_level = log_level_fail
        elif log_level_result is not None:
            result_log_level = log_level_result
        else:
            result_log_level = log_level

        if die_on_error and not result_success:
            if result_log_level < logging.ERROR:
                result_log_level = logging.ERROR
            status_msg += " [FATAL]"

        if text and is_binary:
            status_msg += " [BINARY]"

        if decode_exception:
            # We caught an exception during decoding. We still want to log the result,
            # before re-raising the exception.
            #
            # We don't increase the logging level, because the user requested a special
            # "decode_errors". A decoding error is expected, we just want to log about it
            # (with the level we would have).
            status_msg += " [DECODE_ERROR]"

        if unexpected_binary:
            status_msg += " [UNEXPECTED_BINARY]"
            if result_log_level < logging.ERROR:
                result_log_level = logging.ERROR

        if bin_result.cancelled:
            status_msg += " [CANCELLED]"

        if str_result is not None:
            str_result = str_result.dup_with_forced_success(result_success)
        bin_result = bin_result.dup_with_forced_success(result_success)

        if result_log_level >= 0:
            if log_lineoutput < 0:
                # Line logging is disabled, we print the result now
                with_output = True
            elif result_log_level > log_lineoutput:
                # We printed the lines, but that was at a lower log level than
                # we are to print the result now. Log the output again.
                #
                # This is because here we are likely to log an error, and we
                # want to see the output of the command for why the error
                # happened.
                with_output = True
            else:
                # No need to print the output again.
                with_output = False
            if is_binary:
                # Note that we log the output as binary if either "text=False" or if
                # the output was not valid utf-8. In the latter case, we will still
                # return a string Result (or re-raise decode_exception).
                debug_str = bin_result.debug_str(with_output=with_output)
            else:
                assert str_result is not None
                debug_str = str_result.debug_str(with_output=with_output)

            if log_level >= 0:
                result_kind = "└──>"
            else:
                result_kind = ">"

            logger.log(
                result_log_level,
                f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: {result_kind} {_cmd_to_logstr(real_cmd)}:{status_msg} {debug_str}",
            )

        if decode_exception:
            raise decode_exception

        if die_on_error and not result_success:
            import traceback

            logger.error(
                f"FATAL ERROR. BACKTRACE:\n{''.join(traceback.format_stack())}"
            )
            sys.exit(common.FATAL_EXIT_CODE)

        if str_result is not None:
            return str_result._maybe_intern()
        return bin_result._maybe_intern()

    @abstractmethod
    def _run(
        self,
        *,
        cmd: Union[str, tuple[str, ...]],
        env: Optional[dict[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        handle_line: Optional[Callable[[bool, bytes], None]],
        handle_log: Callable[[int, str], None],
        cancellable: Optional[common.Cancellable],
    ) -> BinResult:
        pass

    @typing.overload
    def run_in_thread(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: typing.Literal[True] = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[Callable[[Result], bool]] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
        start: bool = True,
        add_to_thread_list: bool = False,
        user_data: Any = None,
    ) -> FutureThread[Result]: ...

    @typing.overload
    def run_in_thread(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: typing.Literal[False],
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[Callable[[BinResult], bool]] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
        start: bool = True,
        add_to_thread_list: bool = False,
        user_data: Any = None,
    ) -> FutureThread[BinResult]: ...

    @typing.overload
    def run_in_thread(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: bool = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[
            Union[Callable[[Result], bool], Callable[[BinResult], bool]]
        ] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
        start: bool = True,
        add_to_thread_list: bool = False,
        user_data: Any = None,
    ) -> Union[FutureThread[Result], FutureThread[BinResult]]: ...

    def run_in_thread(
        self,
        cmd: Union[str, Iterable[str]],
        *,
        text: bool = True,
        env: Optional[Mapping[str, Optional[str]]] = None,
        sudo: Optional[bool] = None,
        cwd: Optional[str] = None,
        log_prefix: str = "",
        log_level: Optional[int] = None,
        log_level_result: Optional[int] = None,
        log_level_fail: Optional[int] = None,
        log_lineoutput: Union[bool, int] = False,
        check_success: Optional[
            Union[Callable[[Result], bool], Callable[[BinResult], bool]]
        ] = None,
        die_on_error: bool = False,
        decode_errors: Optional[str] = None,
        cancellable: Optional[common.Cancellable] = None,
        line_callback: Optional[Callable[[bool, bytes], None]] = None,
        start: bool = True,
        add_to_thread_list: bool = False,
        user_data: Any = None,
    ) -> Union[FutureThread[Result], FutureThread[BinResult]]:

        if not isinstance(cmd, str):
            cmd = list(cmd)
        if env is not None:
            env = dict(env)

        thread = FutureThread(
            lambda th: self.run(
                cmd,
                text=text,
                env=env,
                sudo=sudo,
                cwd=cwd,
                log_prefix=log_prefix,
                log_level=log_level,
                log_level_result=log_level_result,
                log_level_fail=log_level_fail,
                log_lineoutput=log_lineoutput,
                check_success=check_success,
                die_on_error=die_on_error,
                cancellable=th.cancellable,
                line_callback=line_callback,
            ),
            cancellable=cancellable,
            start=start,
            user_data=user_data,
        )

        if add_to_thread_list:
            common.thread_list_add(thread)

        return typing.cast(
            Union[FutureThread[Result], FutureThread[BinResult]],
            thread,
        )

    def get_effective_sudo(self, sudo: Optional[bool] = None) -> bool:
        if sudo is None:
            return self.sudo
        return sudo

    def has_sudo(
        self,
        *,
        force_recheck: bool = False,
        log_level: int = logging.DEBUG,
    ) -> bool:
        force_recheck = bool(force_recheck)
        with self._lock:
            if not force_recheck:
                if self._has_sudo is not None:
                    return self._has_sudo
        res = self.run(
            ["sudo", "-n", "whoami"],
            text=False,
            sudo=False,
            log_level_result=log_level,
        )
        has = res.match(
            out=re.compile(b"^[-a-zA-Z_][-a-zA-Z_0-9]*\n$"),
            err=b"",
            returncode=0,
        )
        with self._lock:
            if not force_recheck:
                if self._has_sudo is not None:
                    return self._has_sudo
            self._has_sudo = has
        return has

    def file_exists(
        self,
        path: Union[str, os.PathLike[Any]],
        *,
        cwd: Optional[str] = None,
        sudo: Optional[bool] = None,
        is_dir: Optional[bool] = None,
        log_prefix: str = "",
        log_level: int = -1,
    ) -> bool:
        if is_dir is None:
            flag = "-e"
        elif is_dir:
            flag = "-d"
        else:
            flag = "-f"
        res = self.run(
            ["test", flag, str(path)],
            text=False,
            cwd=cwd,
            sudo=sudo,
            log_prefix=log_prefix,
            log_level_result=log_level,
        )
        return res.success

    def file_remove(
        self,
        path: Union[str, os.PathLike[Any]],
        *,
        cwd: Optional[str] = None,
        sudo: Optional[bool] = None,
        log_prefix: str = "",
        log_level: int = logging.DEBUG,
    ) -> bool:
        res = self.run(
            ["rm", "-rf", str(path)],
            text=False,
            cwd=cwd,
            sudo=sudo,
            log_prefix=log_prefix,
            log_level_result=log_level,
        )
        return res.success


class LocalHost(Host):
    def pretty_str(self) -> str:
        return "localhost"

    def _prepare_run(
        self,
        *,
        cmd: Union[str, Iterable[str]],
        env: Optional[Mapping[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        has_cancellable: bool,
    ) -> tuple[
        Union[str, tuple[str, ...]],
        Optional[dict[str, Optional[str]]],
        Optional[str],
    ]:
        if not sudo:
            return (
                _normalize_cmd(cmd),
                _normalize_env(env),
                cwd,
            )
        return (
            _cmd_with_sudo(cmd=cmd, cwd=cwd, env=env),
            None,
            None,
        )

    def _run(
        self,
        *,
        cmd: Union[str, tuple[str, ...]],
        env: Optional[dict[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        handle_line: Optional[Callable[[bool, bytes], None]],
        handle_log: Callable[[int, str], None],
        cancellable: Optional[common.Cancellable],
    ) -> BinResult:
        full_env: Optional[dict[str, str]] = None
        if env is not None:
            full_env = os.environ.copy()
            for k, v in env.items():
                if v is None:
                    full_env.pop(k, None)
                else:
                    full_env[k] = v

        try:
            # We set stdin to DEVNULL. That is for consistency with
            # RemoteHost, which also does `sh -c {shlex.quote(cmd)} < /dev/null`
            # if the command has a cancellable.
            pr = subprocess.Popen(
                cmd,
                shell=isinstance(cmd, str),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=full_env,
                cwd=cwd,
            )
        except Exception as e:
            # We get an FileNotFoundError if cwd directory does not exist or if
            # the binary does not exist (with shell=False). We get a PermissionError
            # if we don't have permissions.
            #
            # Generally, we don't want to report errors via exceptions, because
            # you won't get the same exception with shell=True. Instead, we
            # only report errors via BinResult().
            #
            # Usually we avoid creating an artificial BinResult. In this case
            # there is no choice.
            return BinResult.new_internal(str(e))

        buffers = (bytearray(), bytearray())

        def _readlines(
            stream: Optional[typing.IO[bytes]],
            *,
            read_all: bool,
        ) -> None:
            assert stream is not None
            if stream is pr.stdout:
                is_stdout = True
            else:
                assert stream is pr.stderr
                is_stdout = False
            while True:
                if read_all:
                    to_read, _, _ = select.select([stream], [], [], 0)
                    if not to_read:
                        return
                b = stream.readline()
                if not b:
                    return
                buffers[is_stdout].extend(b)
                if handle_line is not None:
                    handle_line(is_stdout, b)
                if not read_all:
                    return

        fd_cancellable = cancellable.get_poll_fd() if cancellable else None
        terminate_state = -1
        terminate_kill_at_timestamp = 0.0

        returncode = RETURNCODE_CANCELLED

        while True:
            fds: list[Optional[Union[typing.IO[bytes], int]]] = [pr.stdout, pr.stderr]
            if fd_cancellable is not None:
                fds.append(fd_cancellable)

            select_timeout: Optional[float] = None
            if terminate_state >= 1:
                select_timeout = max(
                    0.0, (terminate_kill_at_timestamp - time.monotonic())
                )

            to_read, _, _ = select.select(fds, [], [], select_timeout)

            for stream in to_read:
                if fd_cancellable is not None and stream == fd_cancellable:
                    fd_cancellable = None
                    terminate_state = 0
                    continue
                _readlines(stream, read_all=False)

            if terminate_state == 0:
                terminate_state = 1
                terminate_kill_at_timestamp = time.monotonic() + TERMINATE_KILL_WAIT
                handle_log(
                    logging.DEBUG,
                    f"cancelled: terminate {'sudo ' if sudo else ''}process {pr.pid}",
                )
                _pr_kill(pr, sudo=sudo, handle_log=handle_log)
            elif terminate_state == 1:
                if time.monotonic() >= terminate_kill_at_timestamp:
                    # The process MUST exit on SIGTERM in time. It is hanging, that is
                    # a bug. The user must not try to cancel processes that refuse to
                    # quit on SIGTERM. The handling here is just trying to recover from
                    # the bad thing that already happend.
                    terminate_state = 2
                    terminate_kill_at_timestamp += TERMINATE_KILL_WAIT / 2.0
                    handle_log(
                        logging.ERROR,
                        f"{'sudo ' if sudo else ''}process {pr.pid} is not quitting in time",
                    )
                    _pr_kill(pr, sudo=sudo, handle_log=handle_log, with_sigkill=True)
            elif terminate_state == 2:
                if time.monotonic() >= terminate_kill_at_timestamp:
                    terminate_state = 3
                    handle_log(
                        logging.ERROR,
                        f"{'sudo ' if sudo else ''}process {pr.pid} cannot be killed and hangs",
                    )
                    break

            if pr.poll() is not None:
                returncode = pr.returncode
                break

        _readlines(pr.stdout, read_all=True)
        _readlines(pr.stderr, read_all=True)

        return BinResult(
            bytes(buffers[True]),
            bytes(buffers[False]),
            returncode,
            cancelled=terminate_state >= 0,
        )

    def file_exists(
        self,
        path: Union[str, os.PathLike[Any]],
        *,
        cwd: Optional[str] = None,
        sudo: Optional[bool] = None,
        is_dir: Optional[bool] = None,
        log_prefix: str = "",
        log_level: int = -1,
    ) -> bool:
        if self.get_effective_sudo(sudo):
            return super().file_exists(
                path,
                cwd=cwd,
                sudo=sudo,
                is_dir=is_dir,
                log_prefix=log_prefix,
                log_level=log_level,
            )

        path = _path_make_absolute(path, cwd=cwd)

        if is_dir is None:
            exists = os.path.exists(path)
            kind = "file"
        elif is_dir:
            exists = os.path.isdir(path)
            kind = "dir"
        else:
            exists = os.path.isfile(path)
            kind = "regular file"

        if log_level >= 0:
            log_id = _unique_log_id()
            logger.log(
                log_level,
                f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: {kind} {repr(path)} exists: {exists}",
            )

        return exists

    def file_remove(
        self,
        path: Union[str, os.PathLike[Any]],
        *,
        cwd: Optional[str] = None,
        sudo: Optional[bool] = None,
        log_prefix: str = "",
        log_level: int = logging.DEBUG,
    ) -> bool:
        if self.get_effective_sudo(sudo):
            return super().file_remove(
                path,
                cwd=cwd,
                sudo=sudo,
                log_prefix=log_prefix,
                log_level=log_level,
            )

        path = _path_make_absolute(path, cwd=cwd)

        try:
            os.remove(path)
        except IsADirectoryError:
            try:
                shutil.rmtree(path, ignore_errors=False)
            except FileNotFoundError:
                result, msg = True, "directory does not exist"
            except Exception as e:
                result, msg = False, f"failed: {e}"
            else:
                result, msg = True, "directory deleted"
        except FileNotFoundError:
            result, msg = True, "file does not exist"
        except Exception as e:
            result, msg = False, f"failed: {e}"
        else:
            result, msg = True, "file deleted"

        if log_level >= 0:
            if result:
                logmsg = f"file {repr(path)} deleted ({msg})"
            else:
                logmsg = f"file {repr(path)} failed to delete ({msg})"
            log_id = _unique_log_id()
            logger.log(
                log_level,
                f"{log_prefix}cmd[{log_id};{self.pretty_str()}]: {logmsg}",
            )

        return result


@dataclass(frozen=True)
class _Login(ABC):
    user: str

    @abstractmethod
    def _login(self, client: "paramiko.SSHClient", host: str) -> None:
        pass


@dataclass(frozen=True, **common.KW_ONLY_DATACLASS)
class AutoLogin(_Login):
    def _login(self, client: "paramiko.SSHClient", host: str) -> None:
        client.connect(
            host,
            username=self.user,
            look_for_keys=True,
            allow_agent=True,
        )


class RemoteHost(Host):
    def __init__(
        self,
        host: str,
        *logins: Union[_Login, str],
        sudo: bool = False,
        login_retry_duration: float = 10 * 60,
    ) -> None:

        logins2 = tuple(
            (AutoLogin(user=login) if isinstance(login, str) else login)
            for login in logins
        )
        if not logins2:
            user = os.environ.get("USER")
            if not user:
                raise ValueError("Unknown USER to login at RemoteHost()")
            logins2 = (AutoLogin(user),)

        import paramiko

        super().__init__(sudo=sudo)
        self.host = host
        self.logins = logins2
        self.login_retry_duration = login_retry_duration
        self._paramiko = paramiko
        self._tlocal = threading.local()

    def pretty_str(self) -> str:
        return f"@{self.host}"

    def _prepare_run(
        self,
        *,
        cmd: Union[str, Iterable[str]],
        env: Optional[Mapping[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        has_cancellable: bool,
    ) -> tuple[
        Union[str, tuple[str, ...]],
        Optional[dict[str, Optional[str]]],
        Optional[str],
    ]:
        cmd = _cmd_to_shell(cmd, cwd=cwd)

        if env:
            # Assume we have a POSIX shell, and we can define variables via `export VAR=...`.
            cmd2 = ""
            for k, v in env.items():
                assert k == shlex.quote(k)
                if v is None:
                    cmd2 += f"unset  -v {k} ; "
                else:
                    cmd2 += f"export {k}={shlex.quote(v)} ; "
            cmd = cmd2 + cmd

        if has_cancellable:
            # We want to abort the process, but paramiko/SSH doesn't directly
            # allow to kill the process.
            #
            # Instead, start a wrapping shell script that reads from STDIN.
            # When STDIN closes, the script kills the command.
            cmd = (
                "bash",
                "-c",
                f"/bin/sh -c {shlex.quote(cmd)} < /dev/null & pid=$!; cat | {{ cat; kill $pid; }} 2> /dev/null & wait $pid",
            )
        else:
            # For consistency with other operations, stdin of the invoked command is /dev/null.
            cmd = f"/bin/sh -c {shlex.quote(cmd)} < /dev/null"

        if sudo:
            cmd = _cmd_with_sudo(cmd=cmd, cwd=None, env=None)

        return cmd, None, None

    def _get_client(
        self,
    ) -> tuple[threading.local, "paramiko.SSHClient", Optional[_Login]]:
        tlocal = self._tlocal
        client = getattr(tlocal, "client", None)
        login: Optional[_Login] = None
        if client is None:
            client = self._paramiko.SSHClient()
            client.set_missing_host_key_policy(self._paramiko.AutoAddPolicy())
            tlocal.client = client
            tlocal.login = None
        else:
            login = tlocal.login
        return tlocal, client, login

    def _clear_client(self) -> None:
        tlocal = self._tlocal
        tlocal.client = None
        tlocal.login = None

    def _ensure_login(
        self,
        *,
        force_new_login: bool,
        start_timestamp: float,
        cancellable: Optional[common.Cancellable],
        handle_log: Callable[[int, str], None],
    ) -> tuple[Optional["paramiko.SSHClient"], bool]:
        tlocal, client, login = self._get_client()

        if login is not None:
            if not force_new_login:
                return client, False

        if start_timestamp == -1.0:
            start_timestamp = time.monotonic()

        end_timestamp = start_timestamp + self.login_retry_duration

        tlocal.login = None

        try_count = 0

        while True:
            for login in self.logins:
                try:
                    login._login(client, self.host)
                except Exception as e:
                    error = e
                else:
                    common.unwrap(client.get_transport()).set_keepalive(1)
                    tlocal.login = login
                    if handle_log is not None:
                        handle_log(
                            logging.DEBUG,
                            f"remote[{self.pretty_str()}]: successfully logged in to {login}",
                        )
                    return client, True

                if try_count == 0:
                    if handle_log is not None:
                        handle_log(
                            logging.DEBUG,
                            f"remote[{self.pretty_str()}]: failed to login to {login}: {error}",
                        )

                if common.Cancellable.is_cancelled(cancellable):
                    return None, False

            try_count += 1

            now = time.monotonic()

            if now >= end_timestamp:
                if handle_log is not None:
                    handle_log(
                        logging.DEBUG,
                        f"remote[{self.pretty_str()}]: failed to login with credentials {self.logins} ({try_count} tries)",
                    )
                return None, False

            time.sleep(min(1.0, end_timestamp - now))

    def _run(
        self,
        *,
        cmd: Union[str, tuple[str, ...]],
        env: Optional[dict[str, Optional[str]]],
        cwd: Optional[str],
        sudo: bool,
        handle_line: Optional[Callable[[bool, bytes], None]],
        handle_log: Callable[[int, str], None],
        cancellable: Optional[common.Cancellable],
    ) -> BinResult:

        assert env is None
        assert cwd is None
        cmd = _cmd_to_shell(cmd)

        bin_cmd: Any = cmd.encode("utf-8", errors="surrogateescape")

        start_timestamp = time.monotonic()
        first_try = True
        while True:
            client, new_login = self._ensure_login(
                start_timestamp=start_timestamp,
                force_new_login=not first_try,
                handle_log=handle_log,
                cancellable=cancellable,
            )
            if client is None:
                if common.Cancellable.is_cancelled(cancellable):
                    return BinResult.CANCELLED
                return BinResult.new_internal(
                    f"failed to login to remote host {self.host}"
                )

            try:
                stdin, stdout, stderr = client.exec_command(bin_cmd)
            except Exception as e:
                if new_login:
                    # We just did a new login and got a failure. Propagate the
                    # error.
                    return BinResult.new_internal(str(e))

                # We had a cached login from earlier. Maybe this was broken
                # and the cause for error now. Retry and force new login.
                first_try = False
                continue

            break

        buffers = (bytearray(), bytearray())
        sources = (stderr, stdout)

        for source in sources:
            source.channel.setblocking(0)

        fds = [s.channel.fileno() for s in sources]
        if cancellable is not None:
            fds.append(cancellable.get_poll_fd())

        def _readlines(*, is_stdout: Optional[bool] = None) -> None:
            if is_stdout is None:
                _readlines(is_stdout=False)
                _readlines(is_stdout=True)
                return
            channel = sources[is_stdout].channel
            buffer = buffers[is_stdout]
            start_idx = len(buffer)
            while True:
                try:
                    if is_stdout:
                        d = channel.recv(32768)
                    else:
                        d = channel.recv_stderr(32768)
                except Exception:
                    d = b""
                if not d:
                    break
                buffer.extend(d)

            if start_idx == len(buffer):
                return

            if handle_line is not None:
                while True:
                    idx = buffer.find(b"\n", start_idx)
                    if idx != -1:
                        line = bytes(buffer[start_idx : idx + 1])
                        start_idx = idx + 1
                    elif start_idx < len(buffer):
                        line = bytes(buffer[start_idx:])
                        start_idx = len(buffer)
                    else:
                        break
                    handle_line(is_stdout, line)

        terminate_state = -1
        terminate_kill_at_timestamp = 0.0

        while True:
            if stdout.channel.exit_status_ready():
                returncode = stdout.channel.recv_exit_status()
                break

            to_read, _, _ = select.select(fds, [], [])
            for fd in to_read:
                if len(fds) == 3 and fd == fds[2]:
                    del fds[2]
                    terminate_state = 0
                    continue
                _readlines(is_stdout=(fd == fds[True]))

            if terminate_state == 0:
                # We signal termination by closing stdin of the remote process.
                stdin.channel.close()
                terminate_kill_at_timestamp = time.monotonic() + (
                    TERMINATE_KILL_WAIT * 1.5
                )
                terminate_state = 1
            elif terminate_state == 1:
                if time.monotonic() >= terminate_kill_at_timestamp:
                    terminate_state = 2
                    handle_log(
                        logging.ERROR, "remote process did not terminate and hangs"
                    )
                    returncode = RETURNCODE_CANCELLED
                    break

        _readlines()

        if terminate_state < 0:
            stdin.channel.close()
        stdout.channel.close()
        stderr.channel.close()

        if terminate_state == 2:
            self._clear_client()

        return BinResult(
            bytes(buffers[True]),
            bytes(buffers[False]),
            returncode,
            cancelled=terminate_state >= 0,
        )


local = LocalHost()


def host_or_local(host: Optional[Host]) -> Host:
    if host is None:
        return local
    return host


if typing.TYPE_CHECKING:
    import paramiko
