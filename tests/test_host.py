import functools
import logging
import os
import pathlib
import pytest
import random
import re
import shlex
import sys
import time

from collections.abc import Mapping
from typing import Any
from typing import Optional
from typing import Union

from ktoolbox import common
from ktoolbox import host

import tstutil


common.log_config_logger(logging.DEBUG, "ktoolbox")


def rnd_one_in(n: int) -> bool:
    return random.randint(0, n - 1) == 0


def rnd_bool() -> bool:
    return rnd_one_in(2)


def rnd_run_extraargs() -> dict[str, Any]:
    args: dict[str, Any] = {}

    r = random.randint(0, 4)
    val: Optional[Union[int, bool]] = None
    if r <= 1:
        val = r == 0
    elif r == 2:
        val = -1
    elif r == 3:
        val = logging.DEBUG
    else:
        val = logging.ERROR
    if val is not None:
        args["log_lineoutput"] = val

    r = random.randint(0, 2)
    if r == 0:
        pass
    elif r == 1:
        args["cancellable"] = None
    else:
        args["cancellable"] = common.Cancellable()
    return args


def rnd_delay(max_delay: float) -> Optional[float]:
    x_percent = max_delay / 10.0
    val = random.random() * (max_delay + 2 * x_percent) - 2 * x_percent
    if val < -x_percent:
        return None
    if val <= 0.0:
        return 0.0
    return min(max_delay, val)


def rnd_sleep(max_delay: float) -> None:
    delay = rnd_delay(max_delay)
    if delay is not None:
        time.sleep(delay)


def rnd_cancel_in_background(delay: float = 0) -> common.Cancellable:
    cancellable = common.Cancellable()

    if random.random() < 0.07:
        cancellable.cancel()
    else:

        def _run(th: common.FutureThread[None]) -> None:
            rnd_sleep(delay)
            cancellable.cancel()

        common.FutureThread(_run, start=True)
    return cancellable


def _system(cmd: str) -> None:
    r = os.system(cmd)
    assert r == 0


@functools.cache
def get_user() -> Optional[str]:
    return os.getenv("USER")


@functools.cache
def has_paramiko() -> bool:
    try:
        import paramiko

        assert paramiko.client
        return True
    except ModuleNotFoundError:
        return False


@functools.cache
def can_ssh_nopass(
    hostname: str,
    user: str,
    *,
    sudo: bool = False,
) -> Optional[host.RemoteHost]:

    if not has_paramiko():
        return None

    import paramiko

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname, username=user, password="")
    except Exception:
        pass
    else:
        _in, _out, _err = client.exec_command("echo -n hello")
        _in.close()
        s_out = _out.read()
        s_err = _err.read()
        rc = _out.channel.recv_exit_status()
        if rc == 0 and s_out == b"hello" and s_err == b"":
            rsh = host.RemoteHost(hostname, host.AutoLogin(user), sudo=sudo)
            res = rsh.run("whoami")
            effective_user = "root" if sudo else user
            if res != host.Result(f"{effective_user}\n", "", 0):
                return None
            return rsh
    return None


def run_local(
    cmd: Union[str, list[str]],
    *,
    text: bool = True,
    env: Optional[Mapping[str, Optional[str]]] = None,
    cwd: Optional[str] = None,
) -> Union[host.Result, host.BinResult]:
    res = host.local.run(cmd, text=text, cwd=cwd, env=env)

    rsh = can_ssh_nopass("localhost", get_user())
    if rsh is not None:
        res2 = rsh.run(cmd, text=text, cwd=cwd, env=env)
        assert res == res2

    return res


def skip_without_paramiko() -> None:
    if not has_paramiko():
        pytest.skip("paramiko module is not available")


def skip_without_ssh_nopass(
    hostname: str = "localhost",
    user: Optional[str] = None,
    *,
    sudo: bool = False,
) -> tuple[str, host.RemoteHost]:
    if user is None:
        user = get_user() or "root"
    skip_without_paramiko()
    rsh = can_ssh_nopass(hostname, user, sudo=sudo)
    if rsh is None:
        pytest.skip(f"cannot ssh to {user}@{hostname} without password")
    return user, rsh


def skip_without_sudo(rsh: host.Host) -> None:
    if not rsh.has_sudo():
        pytest.skip(f"sudo on {rsh.pretty_str()} does not seem to work passwordless")


def test_host_result_bin() -> None:
    res = run_local("echo -n out; echo -n err >&2", text=False)
    assert res == host.BinResult(b"out", b"err", 0)


def test_host_result_surrogateescape() -> None:
    res = host.local.run(
        "echo -n hi", decode_errors="surrogateescape", **rnd_run_extraargs()
    )
    assert res == host.Result("hi", "", 0)

    cmd = ["bash", "-c", "printf $'xx<\\325>'"]

    res_bin = host.local.run(cmd, text=False, **rnd_run_extraargs())
    assert res_bin == host.BinResult(b"xx<\325>", b"", 0)

    res = host.local.run(cmd, decode_errors="surrogateescape")
    assert res == host.Result("xx<\udcd5>", "", 0)
    with pytest.raises(UnicodeEncodeError):
        res.out.encode()
    assert res.out.encode(errors="surrogateescape") == b"xx<\325>"

    res_bin2 = run_local(["bash", "-c", 'printf "xx<\udcd5>"'], text=False)
    assert res_bin2 == host.BinResult(b"xx<\325>", b"", 0)

    res_bin = run_local(["echo", "-n", "xx<\udcd5>"], text=False)
    assert res_bin == host.BinResult(b"xx<\325>", b"", 0)

    cmd2 = b'echo -n "xx<\325>"'.decode(errors="surrogateescape")
    res_bin = host.local.run(cmd2, text=False)
    assert res_bin == host.BinResult(b"xx<\325>", b"", 0)

    t = False
    res_any = host.local.run(cmd2, text=t, **rnd_run_extraargs())
    assert isinstance(res_any, host.BinResult)
    assert res_any == host.BinResult(b"xx<\325>", b"", 0)

    res = host.local.run(cmd2)
    assert res == host.Result("xx<�>", "", 0)

    res = host.local.run(cmd2, decode_errors="surrogateescape")
    assert res == host.Result("xx<\udcd5>", "", 0)

    res_bin = host.local.run(["bash", "-c", cmd2], text=False)
    assert res_bin == host.BinResult(b"xx<\325>", b"", 0)


def test_host_result_str() -> None:
    res = host.local.run("echo -n out; echo -n err >&2", text=True)
    assert res == host.Result("out", "err", 0)

    res = host.local.run("echo -n out; echo -n err >&2", **rnd_run_extraargs())
    assert res == host.Result("out", "err", 0)


def test_host_result_match() -> None:
    res = host.Result("out", "err", 0)

    assert res.match()
    assert res.match(returncode=0)
    assert not res.match(returncode=4)

    assert res.match(out="out")
    assert res.match(out="out", err="err", returncode=0)
    assert res.match(out=re.compile("o"), err="err", returncode=0)
    assert not res.match(out=re.compile("xx"), err="err", returncode=0)

    assert res.match(out=re.compile("."))

    rx = re.compile(b".")
    with pytest.raises(TypeError):
        res.match(out=rx)  # type: ignore

    res_bin = host.BinResult(b"out", b"err", 0)
    assert res_bin.match(out=b"out")
    assert res_bin.match(err=b"err")
    assert res_bin.match(out=re.compile(b"out"))
    assert res_bin.match(out=re.compile(b"^out$"))
    assert res_bin.match(out=re.compile(b"u"))

    assert res_bin.match(out=re.compile(b"."))
    with pytest.raises(TypeError):
        res_bin.match(out=re.compile("."))  # type: ignore


def test_host_various_results() -> None:
    res = host.local.run('printf "foo:\\705x"')
    assert res == host.Result("foo:\ufffdx", "", 0)

    # The result with decode_errors="replace" is the same as if decode_errors
    # is left unspecified. However, the latter case will log an ERROR message
    # when seeing unexpected binary. If you set decode_errors, you expect
    # binary, and no error message is logged.
    res = host.local.run('printf "foo:\\705x"', decode_errors="replace")
    assert res == host.Result("foo:\ufffdx", "", 0)

    res = host.local.run('printf "foo:\\705x"', decode_errors="ignore")
    assert res == host.Result("foo:x", "", 0)

    with pytest.raises(UnicodeDecodeError):
        res = host.local.run('printf "foo:\\705x"', decode_errors="strict")

    res = host.local.run(
        'printf "foo:\\705x"', decode_errors="backslashreplace", **rnd_run_extraargs()
    )
    assert res == host.Result("foo:\\xc5x", "", 0)

    binres = host.local.run('printf "foo:\\705x"', text=False)
    assert binres == host.BinResult(b"foo:\xc5x", b"", 0)


def test_host_check_success() -> None:

    res = host.local.run("echo -n foo", check_success=lambda r: r.success)
    assert res == host.Result("foo", "", 0)
    assert res.success

    res = host.local.run("echo -n foo", check_success=lambda r: r.out != "foo")
    assert res == host.Result("foo", "", 0, forced_success=False)
    assert not res.success

    binres = host.local.run(
        "echo -n foo",
        text=False,
        check_success=lambda r: r.out != b"foo",
        **rnd_run_extraargs(),
    )
    assert binres == host.BinResult(b"foo", b"", 0, forced_success=False)
    assert not binres.success

    res = host.local.run("echo -n foo; exit 74", check_success=lambda r: r.success)
    assert res == host.Result("foo", "", 74)
    assert not res.success
    assert not res

    res = host.local.run("echo -n foo; exit 74", check_success=lambda r: r.out == "foo")
    assert res == host.Result("foo", "", 74, forced_success=True)
    assert res.success

    binres = host.local.run(
        "echo -n foo; exit 74", text=False, check_success=lambda r: r.out == b"foo"
    )
    assert binres == host.BinResult(b"foo", b"", 74, forced_success=True)
    assert binres.success
    assert binres


def test_host_file_exists() -> None:
    assert host.local.file_exists(__file__)
    assert host.Host.file_exists(host.local, __file__)
    assert host.local.file_exists(os.path.dirname(__file__))
    assert host.Host.file_exists(host.local, os.path.dirname(__file__))

    assert host.local.file_exists(pathlib.Path(__file__))
    assert host.Host.file_exists(host.local, pathlib.Path(__file__))


def test_result_typing() -> None:
    host.Result("out", "err", 0)
    host.Result("out", "err", 0, forced_success=True)
    host.BinResult(b"out", b"err", 0)
    host.BinResult(b"out", b"err", 0, forced_success=True)

    if sys.version_info >= (3, 10):
        with pytest.raises(TypeError):
            host.Result("out", "err", 0, True)
        with pytest.raises(TypeError):
            host.BinResult(b"out", b"err", 0, True)
    else:
        host.Result("out", "err", 0, True)
        host.BinResult(b"out", b"err", 0, True)


def test_env() -> None:
    res = host.local.run('echo ">>$FOO<<"', env={"FOO": "xx1"})
    assert res == host.Result(">>xx1<<\n", "", 0)

    res2 = run_local('echo ">>$FOO<<" 1>&2; exit 4', env={"FOO": "xx1"})
    assert res2 == host.Result("", ">>xx1<<\n", 4)


def test_cwd() -> None:
    res = run_local("pwd", cwd="/usr/bin")
    assert res == host.Result("/usr/bin\n", "", 0)

    res = run_local(["pwd"], cwd="/usr/bin")
    assert res == host.Result("/usr/bin\n", "", 0)

    res = host.local.run("pwd", cwd="/usr/bin/does/not/exist")
    assert res.out == ""
    assert res.returncode == host.RETURNCODE_INTERNAL
    assert "/usr/bin/does/not/exist" in res.err

    res = host.local.run("pwd", cwd="/root")
    if res == host.Result("/root\n", "", 0):
        # We have permissions to access the directory.
        pass
    else:
        assert res.out == ""
        assert res.returncode == host.RETURNCODE_INTERNAL
        assert "/root" in res.err


def test_sudo() -> None:
    skip_without_sudo(host.local)

    rsh = host.LocalHost(sudo=True)

    assert rsh.run("whoami") == host.Result("root\n", "", 0)

    assert rsh.run(["whoami"]) == host.Result("root\n", "", 0)

    res = rsh.run('echo ">>$FOO<"', env={"FOO": "xx1"})
    assert res == host.Result(">>xx1<\n", "", 0)

    res = rsh.run(
        ["bash", "-c", 'echo ">>$FOO2<" >&2; exit 55'], env={"FOO2": "xx1", "F1": None}
    )
    assert res == host.Result("", ">>xx1<\n", 55)

    res = rsh.run("pwd", cwd="/usr/bin")
    assert res == host.Result("/usr/bin\n", "", 0)

    res = rsh.run(["pwd"], cwd="/usr/bin")
    assert res == host.Result("/usr/bin\n", "", 0)

    res = rsh.run("echo hi; whoami >&2; pwd", cwd="/usr/bin")
    assert res == host.Result("hi\n/usr/bin\n", "root\n", 0)

    res = rsh.run(["bash", "-c", "echo hi; whoami >&2; pwd"], cwd="/usr/bin")
    assert res == host.Result("hi\n/usr/bin\n", "root\n", 0)

    res = rsh.run("pwd", cwd="/usr/bin/does/not/exist")
    assert res.out == ""
    assert res.returncode == 1
    assert "/usr/bin/does/not/exist" in res.err

    res = rsh.run("pwd", cwd="/root")
    assert res == host.Result("/root\n", "", 0)


def test_remotehost_userdoesnotexist() -> None:
    skip_without_paramiko()

    rsh = host.RemoteHost(
        "localhost", host.AutoLogin(user="userdoesnotexist"), login_retry_duration=0
    )
    res = rsh.run("whoami")
    assert res == host.Result(
        out="",
        err="Host.run(): failed to login to remote host localhost",
        returncode=host.RETURNCODE_INTERNAL,
    )


def test_remotehost_1() -> None:
    user, rsh = skip_without_ssh_nopass()

    res = rsh.run("whoami", **rnd_run_extraargs())
    assert res == host.Result(f"{user}\n", "", 0)

    res = rsh.run(
        'whoami; pwd; echo ">>$FOO<"',
        cwd="/usr",
        env={"FOO": "hi"},
        **rnd_run_extraargs(),
    )
    assert res == host.Result(f"{user}\n/usr\n>>hi<\n", "", 0)


def test_remotehost_sudo() -> None:
    user, rsh = skip_without_ssh_nopass()

    res = rsh.run("whoami", sudo=True, **rnd_run_extraargs())
    if res.success:
        assert res == host.Result("root\n", "", 0)
    else:
        assert res.out == ""
        assert "sudo" in res.err


def _test_cancellable(rsh: host.Host) -> None:
    bin_res = rsh.run("sleep 5", text=False, cancellable=rnd_cancel_in_background(0.1))
    if bin_res != host.BinResult.CANCELLED:
        if isinstance(rsh, host.RemoteHost):
            assert bin_res == host.BinResult(b"", b"", -1, cancelled=True)
        else:
            assert bin_res == host.BinResult(b"", b"", -15, cancelled=True)

    res = rsh.run(
        [
            "bash",
            "-c",
            '_handle_term() { echo hi trapped; test -n "$pid" && kill "$pid"; exit 4; } ; trap _handle_term TERM; sleep 5 & pid=$!; wait $pid;',
        ],
        cancellable=rnd_cancel_in_background(0.1),
    )
    if res != host.Result.CANCELLED:
        if isinstance(rsh, host.RemoteHost):
            assert res == host.Result("", "", -1, cancelled=True)
        else:
            assert res == host.Result(
                "hi trapped\n", "", 4, cancelled=True
            ) or res == host.Result("", "", -15, cancelled=True)

    res = rsh.run("cat", cancellable=rnd_cancel_in_background(0.1))
    if res == host.Result("", "", 0):
        # We redirect /dev/null into the process, which causes cat to quit right
        # way. In this case, the process quit before we cancelled it. We are good.
        pass
    elif res == host.Result.CANCELLED:
        # Cancelled before we could start the process. Also good.
        pass
    else:
        assert res == host.Result("", "", res.returncode, cancelled=True)
        if isinstance(rsh, host.RemoteHost):
            assert res.returncode in [0, -1]
        else:
            assert res.returncode in [0, -15]

    for i in range(3):

        def _run(th: common.FutureThread[host.Result]) -> host.Result:
            rnd_sleep(0.05)
            return rsh.run(
                "sleep 5",
                cancellable=th.cancellable,
            )

        thread = common.FutureThread(_run, start=True)
        rnd_sleep(0.2)
        res = thread.result(cancel=True)
        if res != host.Result.CANCELLED:
            if isinstance(rsh, host.RemoteHost):
                assert res == host.Result("", "", -1, cancelled=True)
            else:
                assert res == host.Result("", "", -15, cancelled=True)


def _test_cat(rsh: host.Host) -> None:
    # cat is curious, since it will quit immediately on /dev/null That is
    # different from many other applications, which would try to read /dev/null
    # and wait forever.
    #
    # To get that behavior consistent, we always spawn our process with
    # /dev/null redirected to it.
    res = rsh.run("cat")
    assert res == host.Result("", "", 0)

    res = rsh.run("cat", cancellable=common.Cancellable())
    assert res == host.Result("", "", 0)

    skip_without_sudo(rsh)

    res = rsh.run("cat", sudo=True)
    assert res == host.Result("", "", 0)

    res = rsh.run("cat", cancellable=common.Cancellable())
    assert res == host.Result("", "", 0)


def test_cat_local() -> None:
    _test_cat(host.local)


def test_cat_remote() -> None:
    user, rsh = skip_without_ssh_nopass()
    _test_cat(rsh)


def test_cancellable_local() -> None:
    _test_cancellable(host.local)


def test_cancellable_remote_1() -> None:
    user, rsh = skip_without_ssh_nopass()
    _test_cancellable(rsh)


def test_cancellable_remote_2() -> None:
    user, rsh = skip_without_ssh_nopass()

    cancellable = common.Cancellable()

    def line_callback(is_stdout: bool, line: bytes) -> None:
        assert is_stdout is True
        assert line == b"foo\n"
        cancellable.cancel()

    thread = common.FutureThread(
        lambda th: rsh.run(
            "echo foo; sleep 13",
            cancellable=cancellable,
            line_callback=line_callback,
        ),
        start=True,
    )
    res = thread.result()
    assert res == host.Result("foo\n", "", -1, cancelled=True)


def test_cancellable_remote_3() -> None:
    user, rsh = skip_without_ssh_nopass()
    skip_without_sudo(rsh)

    res = rsh.run(
        "sleep 14",
        env={"FOO": "foo"},
        cwd="/tmp",
        sudo=True,
        cancellable=rnd_cancel_in_background(0.1),
    )
    if res != host.Result.CANCELLED:
        assert res == host.Result("", "", -1, cancelled=True)


def _test_cancellable_1(rsh: host.Host) -> None:
    cancellable = common.Cancellable()

    def _line_callback(is_stdout: bool, line: bytes) -> None:
        if line == b"done\n":
            cancellable.cancel()

    thread = common.FutureThread(
        lambda th: rsh.run(
            'pwd; whoami; echo ">$FOO<"; /bin/echo done; sleep 14',
            env={"FOO": "foo"},
            cwd="/tmp",
            sudo=True,
            cancellable=cancellable,
            line_callback=_line_callback,
        ),
        start=True,
    )
    res = thread.result()
    assert res == host.Result(
        "/tmp\nroot\n>foo<\ndone\n",
        "",
        -15 if isinstance(rsh, host.LocalHost) else -1,
        cancelled=True,
    )


def test_cancellable_1_local() -> None:
    skip_without_sudo(host.local)
    _test_cancellable_1(host.local)


def test_cancellable_1_remote() -> None:
    user, rsh = skip_without_ssh_nopass()
    skip_without_sudo(rsh)
    _test_cancellable_1(rsh)


def _test_file_remove(rsh: host.Host, tmp_path: pathlib.Path) -> None:
    def _cleanup_basedir(basedir: str) -> None:
        cmd = ["rm", "-rf", basedir]
        if host.local.has_sudo():
            host.local.run(cmd, sudo=True)
        else:
            host.local.run(cmd, sudo=False)
        assert not os.path.exists(basedir)

    def _file_exists(
        rsh: host.Host,
        path: str,
        *,
        sudo: Optional[bool] = None,
        is_dir: Optional[bool] = None,
    ) -> bool:
        assert path
        assert path[0] == "/"

        if rnd_bool():
            file = path
            cwd = None
        else:
            file = os.path.basename(path)
            cwd = os.path.dirname(path)

        return rsh.file_exists(
            file,
            cwd=cwd,
            sudo=sudo,
            is_dir=is_dir,
        )

    def _file_remove(
        rsh: host.Host,
        path: str,
        *,
        sudo: Optional[bool] = None,
    ) -> bool:
        assert path
        assert path[0] == "/"

        if rnd_bool():
            file = path
            cwd = None
        else:
            file = os.path.basename(path)
            cwd = os.path.dirname(path)

        return rsh.file_remove(
            file,
            cwd=cwd,
            sudo=sudo,
        )

    def _assert_exists(
        path: str,
        *,
        rsh: Optional[host.Host] = None,
        is_dir: Optional[bool] = None,
        sudo: Optional[bool] = None,
    ) -> None:
        if sudo is not None and sudo:
            if os.path.exists(path):
                if is_dir is not None:
                    if is_dir:
                        assert os.path.isdir(path)
                    else:
                        assert os.path.isfile(path)
        else:
            assert os.path.exists(path)
            if is_dir is not None:
                if is_dir:
                    assert os.path.isdir(path)
                else:
                    assert os.path.isfile(path)
        if rnd_one_in(5):
            assert _file_exists(
                host.local,
                path,
                sudo=sudo,
                is_dir=random.choice([None, is_dir]),
            )
        if rsh is not None and rsh is not host.local and rnd_one_in(5):
            assert _file_exists(
                rsh,
                path,
                sudo=sudo,
                is_dir=random.choice([None, is_dir]),
            )

    def _assert_not_exists(
        path: str,
        *,
        rsh: Optional[host.Host] = None,
        allow_with_sudo: bool = True,
        allow_without_sudo: bool = True,
    ) -> None:
        assert not os.path.exists(path)

        def _effective_sudo(rsh: host.Host) -> Optional[bool]:
            if allow_without_sudo and allow_with_sudo:
                return random.choice([None, False, True])
            if allow_without_sudo:
                if not rsh.sudo:
                    return random.choice([None, False])
                return False
            if rsh.sudo:
                return random.choice([None, True])
            return True

        if rnd_one_in(5):
            assert not _file_exists(
                host.local,
                path,
                sudo=_effective_sudo(host.local),
                is_dir=random.choice([None, False, True]),
            )
        if rsh is not None and rsh is not host.local and rnd_one_in(5):
            assert not _file_exists(
                rsh,
                path,
                sudo=_effective_sudo(rsh),
                is_dir=random.choice([None, False, True]),
            )

    def _run_one(basedir: str) -> None:
        _system(f"mkdir {shlex.quote(basedir)}")
        _assert_exists(basedir, rsh=rsh, is_dir=True)

        path = os.path.join(basedir, f"name-{i}")

        is_dir = rnd_bool()
        if is_dir:
            _system(f"mkdir {shlex.quote(path)}")
        else:
            _system(f"touch {shlex.quote(path)}")
        _assert_exists(path, rsh=rsh, is_dir=is_dir)

        is_dir_full = is_dir and rnd_bool()
        if is_dir_full:
            path2 = os.path.join(path, "file1")
            _system(f"touch {shlex.quote(path2)}")
            _assert_exists(path2, rsh=rsh, is_dir=False)

        _system(f"chmod 700 {shlex.quote(os.path.dirname(path))}")

        requires_sudo = host.local.has_sudo() and rnd_bool()
        if requires_sudo:
            assert host.local.run(
                ["chown", "root:root", os.path.dirname(path)],
                sudo=True,
            )

        _assert_exists(os.path.dirname(path), rsh=rsh, is_dir=True)

        if not requires_sudo or os.path.exists(path):
            _assert_exists(path, rsh=rsh, is_dir=is_dir)

        use_sudo = random.choice([None, False, True])

        success = _file_remove(
            rsh,
            path,
            sudo=use_sudo,
        )

        if not success:
            if requires_sudo and not os.path.exists(path):
                _assert_not_exists(path, rsh=rsh, allow_with_sudo=False)
            else:
                _assert_exists(path, rsh=rsh, is_dir=is_dir, sudo=requires_sudo)
            if rsh.has_sudo():
                assert not rsh.get_effective_sudo(use_sudo)
                assert requires_sudo
            else:
                assert rsh.get_effective_sudo(use_sudo) or requires_sudo
        else:
            _assert_not_exists(path, rsh=rsh)

    basedir = str(tmp_path / "base")
    _cleanup_basedir(basedir)
    for i in range(5):
        try:
            _run_one(basedir)
        finally:
            _cleanup_basedir(basedir)


def test_file_remove_local_1(tmp_path: pathlib.Path) -> None:
    _test_file_remove(host.local, tmp_path)


def test_file_remove_local_2(tmp_path: pathlib.Path) -> None:
    rsh = host.LocalHost(sudo=True)
    skip_without_sudo(rsh)
    _test_file_remove(rsh, tmp_path)


def test_file_remove_remote_1(tmp_path: pathlib.Path) -> None:
    user, rsh = skip_without_ssh_nopass()
    _test_file_remove(rsh, tmp_path)


def test_file_remove_remote_2(tmp_path: pathlib.Path) -> None:
    user, rsh = skip_without_ssh_nopass(sudo=True)
    skip_without_sudo(rsh)
    _test_file_remove(rsh, tmp_path)


def _test_run_in_thread(rsh: host.Host) -> None:

    with tstutil.maybe_thread_pool_executor() as executor:

        th = rsh.run_in_thread("echo hi", executor=executor, add_to_thread_list=True)
        assert th.is_started
        assert common.thread_list_get() == [th]
        assert th.result() == host.Result("hi\n", "", 0)
        common.thread_list_join_all()
        assert common.thread_list_get() == []

    with tstutil.maybe_thread_pool_executor() as executor:
        th = rsh.run_in_thread(
            "echo foo >&2 ; exit 7",
            start=False,
            check_success=lambda r: r.returncode == 7,
            executor=executor,
        )
        assert not th.is_started
        th.start()
        assert th.is_started
        th.start()
        assert th.result() == host.Result("", "foo\n", 7, forced_success=True)

    with tstutil.maybe_thread_pool_executor() as executor:
        th2 = rsh.run_in_thread(
            "echo hi; exit 6",
            text=False,
            executor=executor,
        )
        assert th2.result() == host.BinResult(b"hi\n", b"", 6)

    with tstutil.maybe_thread_pool_executor() as executor:
        th = rsh.run_in_thread(
            "sleep 10000",
            start=False,
            executor=executor,
            add_to_thread_list=True,
        )
        assert common.thread_list_get() == [th]
        assert not th.is_started
        with pytest.raises(RuntimeError):
            th.poll()
        th.start()
        assert th.poll() is None
        if rnd_bool():
            common.thread_list_join_all(cancel=True)
        else:
            th.cancellable.cancel()
        r = th.result()
        common.thread_list_join_all()
        assert common.thread_list_get() == []
    if r == host.Result.CANCELLED:
        pass
    else:
        if isinstance(rsh, host.LocalHost):
            assert r == host.Result("", "", -15, cancelled=True)
        else:
            assert r == host.Result("", "", -1, cancelled=True)
    assert r is th.poll()
    assert r is th.result()


def test_run_in_thread_local() -> None:
    _test_run_in_thread(host.local)


def test_run_in_thread_ssh() -> None:
    user, rsh = skip_without_ssh_nopass(sudo=True)
    _test_run_in_thread(rsh)
