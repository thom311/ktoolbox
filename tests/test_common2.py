import os
import pytest
import random
import shlex
import string
import threading
import time

from typing import Optional

from ktoolbox import common
from ktoolbox import host
from ktoolbox import tstutil


def test_future_thread() -> None:

    with tstutil.maybe_thread_pool_executor() as executor:
        thread = common.FutureThread(
            lambda th: host.local.run("echo hi"),
            start=True,
            executor=executor,
        )
        assert thread.result() == host.Result("hi\n", "", 0)

    with tstutil.maybe_thread_pool_executor() as executor:
        thread = common.FutureThread(
            lambda th: host.local.run("sleep 10000", cancellable=th.cancellable),
            start=True,
            executor=executor,
        )
        assert thread.poll() is None
        thread.cancellable.cancel()

        end_time = time.monotonic() + 5.0
        while True:
            r = thread.poll()
            if r is not None:
                assert (
                    r
                    == host.Result(
                        out="",
                        err="",
                        returncode=-15,
                        cancelled=True,
                    )
                    or r == host.Result.CANCELLED
                )
                assert r is thread.result()
                break
            assert time.monotonic() < end_time

    th = common.FutureThread(lambda th: None, start=True)
    assert th.result_full() == (True, None)


def test_future_thread_async() -> None:
    import asyncio

    async def test1() -> None:
        cancellable = common.Cancellable()

        def _cancel_in_background(cancellable: common.Cancellable) -> None:
            def _background_cancel() -> None:
                time.sleep(random.uniform(0, 0.1))
                cancellable.cancel()

            threading.Thread(
                target=_background_cancel,
                daemon=True,
            ).start()

        def _assert_result_is_cancelled(result: Optional[host.Result]) -> None:
            assert result is host.Result.CANCELLED or result == host.Result(
                out="",
                err="",
                returncode=-15,
                cancelled=True,
            )

        def _assert_ft_running_and_cancel(ft: common.FutureThread[host.Result]) -> None:
            assert ft.poll() is None
            _assert_result_is_cancelled(ft.result(cancel=True))

        def _assert_ft_is_cancelled(ft: common.FutureThread[host.Result]) -> None:
            _assert_result_is_cancelled(ft.poll())

        _cancel_in_background(cancellable)
        ft = host.local.run_in_thread("sleep 10000", cancellable=cancellable)
        result = await ft
        _assert_result_is_cancelled(result)

        ft = host.local.run_in_thread("echo hi")
        result = await ft
        assert result == host.Result(out="hi\n", err="", returncode=0)

        ft = host.local.run_in_thread("sleep 1000")
        with pytest.raises(asyncio.TimeoutError):
            await ft.async_result(timeout=0.01)
        _assert_ft_is_cancelled(ft)

        ft = host.local.run_in_thread("sleep 1000")
        with pytest.raises(asyncio.TimeoutError):
            await ft.async_result(timeout=0.01, cancel=False)
        _assert_ft_running_and_cancel(ft)

        ft = host.local.run_in_thread("sleep 1000")
        task = asyncio.create_task(ft.async_result())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_ft_is_cancelled(ft)

        ft = host.local.run_in_thread("sleep 1000")
        task = asyncio.create_task(ft.async_result(cancel=False))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_ft_running_and_cancel(ft)

    asyncio.run(test1())


def test_thread_list() -> None:
    lst = (host.local.run_in_thread("echo"),)
    common.thread_list_cancel(threads=lst)
    common.thread_list_join_all(threads=lst, cancel=True)
    common.thread_list_join_all(threads=lst)

    lst2: tuple[threading.Thread, ...] = ()
    common.thread_list_cancel(threads=lst2)
    common.thread_list_join_all(threads=lst2)


def test_sed_escape_repl() -> None:
    assert common.sed_escape_repl("") == ""
    assert common.sed_escape_repl("abc\\d/x&dd") == "abc\\\\d\\/x\\&dd"

    def _test_pattern(replacement: str) -> None:
        cmd = ["sed", f"s/PATTERN/{common.sed_escape_repl(replacement)}/"]
        ret = host.local.run(f"printf '%s' '[PATTERN]' | {shlex.join(cmd)}")
        assert ret == host.Result(f"[{replacement}]", "", 0)

        arg = f"s//{common.sed_escape_repl(replacement)}/"
        assert os.system(f"sed {shlex.quote(arg)} /dev/null") == 0

    _test_pattern("")
    _test_pattern("x")
    _test_pattern("x\n")
    _test_pattern("simple")
    _test_pattern("with/slash")
    _test_pattern("with&and&multiple&")
    _test_pattern("back\\slash")
    _test_pattern("mix\\&/\\\\")
    _test_pattern("ends_with_backslash\\\\")
    _test_pattern("newline\ninside")
    _test_pattern("multiple\nnewlines\nhere")
    _test_pattern("weird /&\\ combination")
    _test_pattern("\n&/\\\n&/\\\n&/\\")
    _test_pattern("unicode: üñîçødë & / \\")
    _test_pattern("\1\2\3\4\5\6\7")
    _test_pattern("tabs\tand\ttabs")
    _test_pattern("quotes'\"`")
    _test_pattern("spaces and    multiple   spaces")
    _test_pattern("mix1234567890!@#$%^&*()_+-=[]{}|;:,<.>/?")
    _test_pattern("edge-case\\&/\n\\&/\n")
    _test_pattern("repeated_specials" * 5)
    _test_pattern("\\" * 10 + "&/" * 5 + "\n" * 3)
    _test_pattern("emoji 😀 😁 😂 & / \\ \n")
    _test_pattern("control_chars" + "".join(chr(i) for i in range(1, 32)))
    for i in range(10):
        _test_pattern(
            "".join(
                random.choice(string.ascii_letters + string.digits + "\\&/\n")
                for _ in range(50)
            )
        )
