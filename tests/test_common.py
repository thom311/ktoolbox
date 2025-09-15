import argparse
import dataclasses
import io
import json
import logging
import os
import pathlib
import pytest
import random
import re
import socket
import sys
import threading
import time
import typing

from collections.abc import Iterable
from enum import Enum
from typing import Any
from typing import Optional
from typing import Union

from ktoolbox import common
from ktoolbox import host

from ktoolbox.common import enum_convert
from ktoolbox.common import enum_convert_list
from ktoolbox.common import serialize_enum
from ktoolbox.common import StructParsePopContext
from ktoolbox.common import StructParseParseContext

import tstutil


class ReachedError(Exception):
    pass


class TstTestType(Enum):
    IPERF_TCP = 1
    IPERF_UDP = 2
    HTTP = 3
    NETPERF_TCP_STREAM = 4
    NETPERF_TCP_RR = 5


class TstPodType(Enum):
    NORMAL = 1
    SRIOV = 2
    HOSTBACKED = 3


def test_str_to_bool() -> None:
    assert common.str_to_bool(True) is True
    assert common.str_to_bool(False) is False

    ERR = object()
    DEF = object()

    # falsy
    assert common.str_to_bool("0") is False
    assert common.str_to_bool("n") is False
    assert common.str_to_bool("no") is False
    assert common.str_to_bool("false") is False

    # truthy
    assert common.str_to_bool("1") is True
    assert common.str_to_bool("y") is True
    assert common.str_to_bool("yes") is True
    assert common.str_to_bool("true") is True

    assert common.str_to_bool("bogus", None) is None
    assert common.str_to_bool("default", None) is None
    assert common.str_to_bool("bogus", ERR) is ERR
    assert common.str_to_bool("default", ERR) is ERR

    assert common.str_to_bool("bogus", ERR, on_default=DEF) is ERR
    assert common.str_to_bool("default", ERR, on_default=DEF) is DEF

    with pytest.raises(TypeError):
        common.str_to_bool("default", ERR, on_error=DEF)  # type: ignore

    with pytest.raises(ValueError):
        assert common.str_to_bool("bogus", on_default=DEF)
    assert common.str_to_bool("bogus", on_default=DEF, on_error=ERR) is ERR
    assert common.str_to_bool("bogus", on_error=ERR) is ERR
    assert common.str_to_bool("default", on_default=DEF) is DEF
    assert common.str_to_bool("default", on_default=DEF, on_error=ERR) is DEF
    assert common.str_to_bool("default", on_error=ERR) is ERR

    # edge cases
    with pytest.raises(ValueError):
        common.str_to_bool(None)
    assert common.str_to_bool(None, on_default=DEF) is DEF
    with pytest.raises(ValueError):
        common.str_to_bool(0, on_default=DEF)  # type: ignore

    obj = object()

    assert common.str_to_bool("True", on_default=False) is True

    assert common.str_to_bool("", on_default=obj) is obj

    with pytest.raises(ValueError):
        common.str_to_bool("")

    assert common.str_to_bool(None, on_default=obj) is obj
    assert common.str_to_bool("", on_default=obj) is obj
    assert common.str_to_bool(" DEFAULT ", on_default=obj) is obj
    assert common.str_to_bool(" -1 ", on_default=obj) is obj

    assert common.str_to_bool(" -1 ", on_default=DEF) is DEF
    assert common.str_to_bool(" -1 ", on_error=ERR) is ERR

    assert common.str_to_bool("", on_default=DEF, on_error=ERR) is DEF
    assert common.str_to_bool("", on_error=ERR) is ERR

    with pytest.raises(ValueError):
        common.str_to_bool("bogus")

    assert common.str_to_bool("bogus", on_error=ERR) is ERR
    assert common.str_to_bool("bogus", on_default=DEF, on_error=obj) is obj

    assert common.bool_to_str(True) == "true"
    assert common.bool_to_str(False) == "false"
    assert common.bool_to_str(True, format="true") == "true"
    assert common.bool_to_str(False, format="true") == "false"
    assert common.bool_to_str(True, format="yes") == "yes"
    assert common.bool_to_str(False, format="yes") == "no"
    assert common.bool_to_str(True, format="1") == "1"
    assert common.bool_to_str(False, format="1") == "0"
    assert common.bool_to_str(True, format="on") == "on"
    assert common.bool_to_str(False, format="on") == "off"
    with pytest.raises(ValueError):
        common.bool_to_str(False, format="bogus")

    if sys.version_info >= (3, 11):
        typing.assert_type(common.str_to_bool("true"), bool)
        typing.assert_type(common.str_to_bool("true", on_error=None), bool | None)
        typing.assert_type(common.str_to_bool("true", on_default=None), bool | None)
        typing.assert_type(
            common.str_to_bool("true", on_error=None, on_default=1), bool | None | int
        )
        typing.assert_type(
            common.str_to_bool("true", on_error=2, on_default=1), bool | int
        )
        typing.assert_type(
            common.str_to_bool("true", on_error=True, on_default="1"), bool | str
        )
        typing.assert_type(
            common.str_to_bool("true", on_error=True, on_default="1"), bool | str
        )


def test_enum_convert() -> None:
    assert enum_convert(TstTestType, "IPERF_TCP") == TstTestType.IPERF_TCP
    assert enum_convert(TstPodType, 1) == TstPodType.NORMAL
    assert enum_convert(TstPodType, "1 ") == TstPodType.NORMAL
    assert enum_convert(TstPodType, " normal") == TstPodType.NORMAL
    with pytest.raises(ValueError):
        enum_convert(TstTestType, "Not_in_enum")
    with pytest.raises(ValueError):
        enum_convert(TstTestType, 10000)

    assert enum_convert_list(TstTestType, [1]) == [TstTestType.IPERF_TCP]
    assert enum_convert_list(TstTestType, [TstTestType.IPERF_TCP]) == [
        TstTestType.IPERF_TCP
    ]
    assert enum_convert_list(TstTestType, ["iperf-tcp"]) == [TstTestType.IPERF_TCP]
    assert enum_convert_list(TstTestType, ["iperf_tcp-1,3", 2]) == [
        TstTestType.IPERF_TCP,
        TstTestType.HTTP,
        TstTestType.IPERF_UDP,
    ]

    assert enum_convert_list(TstTestType, "*") == list(TstTestType)
    assert enum_convert_list(TstTestType, "1-100") == list(TstTestType)
    with pytest.raises(ValueError):
        enum_convert_list(TstTestType, "*", default_range=None)
    with pytest.raises(ValueError):
        enum_convert_list(TstTestType, "1-*")
    assert enum_convert_list(
        TstTestType, "*", default_range=[TstTestType.HTTP, TstTestType.IPERF_TCP]
    ) == [TstTestType.HTTP, TstTestType.IPERF_TCP]

    class E1(Enum):
        Vm4 = -4
        V1 = 1
        V1b = 1
        v2 = 2
        V2 = 3
        V_7 = 10
        V_3 = 6
        v_3 = 7
        V5 = 5

    assert enum_convert(E1, "v5") == E1.V5
    assert enum_convert(E1, "v2") == E1.v2
    assert enum_convert(E1, "V2") == E1.V2
    assert enum_convert(E1, "v_3") == E1.v_3
    assert enum_convert(E1, "V_3") == E1.V_3
    assert enum_convert(E1, "V_7") == E1.V_7
    assert enum_convert(E1, "V-7") == E1.V_7
    with pytest.raises(ValueError):
        assert enum_convert(E1, "v-3") == E1.v_3
    with pytest.raises(ValueError):
        assert enum_convert(E1, "V-3") == E1.V_3
    assert enum_convert(E1, "10") == E1.V_7
    assert enum_convert(E1, "-4") == E1.Vm4
    assert enum_convert(E1, "1") == E1.V1


def test_serialize_enum() -> None:
    # Test with enum value
    assert serialize_enum(TstTestType.IPERF_TCP) == "IPERF_TCP"

    # Test with a dictionary containing enum values
    data = {
        "test_type": TstTestType.IPERF_UDP,
        "pod_type": TstPodType.SRIOV,
        "other_key": "some_value",
    }
    serialized_data = serialize_enum(data)
    assert serialized_data == {
        "test_type": "IPERF_UDP",
        "pod_type": "SRIOV",
        "other_key": "some_value",
    }

    # Test with a list containing enum values
    data_list = [TstTestType.HTTP, TstPodType.HOSTBACKED]
    serialized_list = serialize_enum(data_list)
    assert serialized_list == ["HTTP", "HOSTBACKED"]

    class TstTestCaseType(Enum):
        POD_TO_HOST_SAME_NODE = 5

    class TstConnectionMode(Enum):
        EXTERNAL_IP = 1

    class TstNodeLocation(Enum):
        SAME_NODE = 1

    # Test with nested structures
    nested_data = {
        "nested_dict": {"test_case_id": TstTestCaseType.POD_TO_HOST_SAME_NODE},
        "nested_list": [TstConnectionMode.EXTERNAL_IP, TstNodeLocation.SAME_NODE],
    }
    serialized_nested_data = serialize_enum(nested_data)
    assert serialized_nested_data == {
        "nested_dict": {"test_case_id": "POD_TO_HOST_SAME_NODE"},
        "nested_list": ["EXTERNAL_IP", "SAME_NODE"],
    }

    # Test with non-enum value
    assert serialize_enum("some_string") == "some_string"
    assert serialize_enum(123) == 123


def test_strict_dataclass() -> None:
    @common.strict_dataclass
    @dataclasses.dataclass
    class C2:
        a: str
        b: int
        c: typing.Optional[str] = None

    C2("a", 5)
    C2("a", 5, None)
    C2("a", 5, "")
    with pytest.raises(TypeError):
        C2("a", "5")  # type: ignore
    with pytest.raises(TypeError):
        C2(3, 5)  # type: ignore
    with pytest.raises(TypeError):
        C2("a", 5, [])  # type: ignore

    @common.strict_dataclass
    @dataclasses.dataclass
    class C3:
        a: list[str]

    C3([])
    C3([""])
    with pytest.raises(TypeError):
        C3(1)  # type: ignore
    with pytest.raises(TypeError):
        C3([1])  # type: ignore
    with pytest.raises(TypeError):
        C3(None)  # type: ignore

    @common.strict_dataclass
    @dataclasses.dataclass
    class C4:
        a: typing.Optional[list[str]]

    C4(None)

    @common.strict_dataclass
    @dataclasses.dataclass
    class C5:
        a: typing.Optional[list[dict[str, str]]] = None

    C5(None)
    C5([])
    with pytest.raises(TypeError):
        C5([1])  # type: ignore
    C5([{}])
    C5([{"a": "b"}])
    C5([{"a": "b"}, {}])
    C5([{"a": "b"}, {"c": "", "d": "x"}])
    with pytest.raises(TypeError):
        C5([{"a": None}])  # type: ignore

    @common.strict_dataclass
    @dataclasses.dataclass
    class C6:
        a: typing.Optional[tuple[str, str]] = None

    C6()
    C6(None)
    C6(("a", "b"))
    with pytest.raises(TypeError):
        C6(1)  # type: ignore
    with pytest.raises(TypeError):
        C6(("a",))  # type: ignore
    with pytest.raises(TypeError):
        C6(("a", "b", "c"))  # type: ignore
    with pytest.raises(TypeError):
        C6(("a", 1))  # type: ignore

    @common.strict_dataclass
    @dataclasses.dataclass(frozen=True)
    class TstPodInfo:
        name: str
        pod_type: TstPodType
        is_tenant: bool
        index: int

    @common.strict_dataclass
    @dataclasses.dataclass
    class C7:
        addr_info: list[TstPodInfo]

        def _post_check(self) -> None:
            pass

    with pytest.raises(TypeError):
        C7(None)  # type: ignore
    C7([])
    C7([TstPodInfo("name", TstPodType.NORMAL, True, 5)])
    with pytest.raises(TypeError):
        C7([TstPodInfo("name", TstPodType.NORMAL, True, 5), None])  # type: ignore

    @common.strict_dataclass
    @dataclasses.dataclass
    class C8:
        a: str

        def _post_check(self) -> None:
            if self.a == "invalid":
                raise ValueError("_post_check() failed")

    with pytest.raises(TypeError):
        C8(None)  # type: ignore
    C8("hi")
    with pytest.raises(ValueError):
        C8("invalid")

    @common.strict_dataclass
    @dataclasses.dataclass
    class C9:
        a: "str"

    with pytest.raises(NotImplementedError):
        C9("foo")

    @common.strict_dataclass
    @dataclasses.dataclass
    class C10:
        x: float

    C10(1.0)
    with pytest.raises(TypeError):
        C10(1)


def test_dataclass_tofrom_dict() -> None:
    @common.strict_dataclass
    @dataclasses.dataclass
    class C1:
        foo: int
        str: typing.Optional[str]

    c1 = C1(1, "str")
    d1 = common.dataclass_to_dict(c1)
    assert c1 == common.dataclass_from_dict(C1, d1)

    @common.strict_dataclass
    @dataclasses.dataclass
    class C2:
        enum_val: TstTestType
        c1_opt: typing.Optional[C1]
        c1_opt_2: typing.Optional[C1]
        c1_list: list[C1]

    c2 = C2(TstTestType.IPERF_UDP, C1(2, "2"), None, [C1(3, "3"), C1(4, "4")])
    d2 = common.dataclass_to_dict(c2)
    assert (
        json.dumps(d2)
        == '{"enum_val": "IPERF_UDP", "c1_opt": {"foo": 2, "str": "2"}, "c1_opt_2": null, "c1_list": [{"foo": 3, "str": "3"}, {"foo": 4, "str": "4"}]}'
    )
    assert c2 == common.dataclass_from_dict(C2, d2)

    @common.strict_dataclass
    @dataclasses.dataclass
    class C10:
        x: float

    assert common.dataclass_to_dict(C10(1.0)) == {"x": 1.0}

    c10 = C10(1.0)
    assert type(c10.x) is float
    common.dataclass_check(c10)
    c10.x = 1
    assert type(c10.x) is int
    with pytest.raises(TypeError):
        common.dataclass_check(c10)
    assert common.dataclass_to_dict(c10) == {"x": 1}

    assert common.dataclass_from_dict(C10, {"x": 1.0}) == c10
    assert common.dataclass_from_dict(C10, {"x": 1.0}) == C10(1.0)
    assert common.dataclass_from_dict(C10, {"x": 1}) == c10
    assert common.dataclass_from_dict(C10, {"x": 1}) == C10(1.0)
    assert type(common.dataclass_from_dict(C10, {"x": 1}).x) is float
    assert type(common.dataclass_from_dict(C10, {"x": 1.0}).x) is float
    assert type(c10.x) is int
    assert type(C10(1.0).x) is float

    @common.strict_dataclass
    @dataclasses.dataclass
    class C11:
        lst: tuple[C10, ...]

    c11 = C11(lst=(c10,))

    assert common.dataclass_to_dict(c11) == {"lst": ({"x": 1},)}
    assert common.dataclass_to_json(c11) == '{"lst": [{"x": 1}]}'
    assert common.dataclass_from_dict(C11, {"lst": [{"x": 1}]}) == c11
    assert common.dataclass_from_dict(C11, {"lst": ({"x": 1},)}) == c11
    assert common.dataclass_from_dict(C11, json.loads('{"lst": [{"x": 1}]}')) == c11


def test_iter_get_first() -> None:

    lst: list[int]

    lst = []
    v2a = common.iter_get_first(lst)
    if sys.version_info >= (3, 11):
        typing.assert_type(v2a, Optional[int])
    assert v2a is None

    lst = [123]
    v2b = common.iter_get_first(lst)
    if sys.version_info >= (3, 11):
        typing.assert_type(v2b, Optional[int])
    assert v2b == 123

    lst = [12, 13]
    v2c = common.iter_get_first(lst)
    if sys.version_info >= (3, 11):
        typing.assert_type(v2c, Optional[int])
    assert v2c == 12

    lst = []
    v3a = common.iter_get_first(lst, unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v3a, Optional[int])
    assert v3a is None

    lst = [123]
    v3b = common.iter_get_first(lst, unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v3b, Optional[int])
    assert v3b == 123

    lst = [12, 13]
    v3c = common.iter_get_first(lst, unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v3c, Optional[int])
    assert v3c is None

    lst = []
    v5a = common.iter_get_first(lst, force_unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v5a, Optional[int])
    assert v5a is None

    lst = [123]
    v5b = common.iter_get_first(lst, force_unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v5b, Optional[int])
    assert v5b == 123

    with pytest.raises(ValueError):
        lst = [12, 13]
        v5c = common.iter_get_first(lst, force_unique=True)
        if sys.version_info >= (3, 11):
            typing.assert_type(v5c, Optional[int])
        assert False

    with pytest.raises(ValueError):
        lst = []
        v1a = common.iter_get_first(lst, unique=True, force_unique=True)
        if sys.version_info >= (3, 11):
            typing.assert_type(v1a, int)
        assert False

    lst = [101]
    v1b = common.iter_get_first(lst, unique=True, force_unique=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v1b, int)
    assert v1b == 101

    with pytest.raises(ValueError):
        lst = [102, 103]
        v1c = common.iter_get_first(lst, unique=True, force_unique=True)
        if sys.version_info >= (3, 11):
            typing.assert_type(v1c, int)
        assert False

    with pytest.raises(ValueError):
        lst = []
        v6a = common.iter_get_first(lst, single=True)
        if sys.version_info >= (3, 11):
            typing.assert_type(v6a, int)
        assert False

    lst = [101]
    v6b = common.iter_get_first(lst, single=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(v6b, int)
    assert v6b == 101

    with pytest.raises(ValueError):
        lst = [102, 103]
        v6c = common.iter_get_first(lst, single=True)
        if sys.version_info >= (3, 11):
            typing.assert_type(v6c, int)
        assert False


def test_kw_only() -> None:
    common.StructParseBaseNamed(yamlpath="yamlpath", yamlidx=0, name="name")
    if sys.version_info >= (3, 10):
        assert common.KW_ONLY_DATACLASS == {"kw_only": True}
        with pytest.raises(TypeError):
            common.StructParseBaseNamed("yamlpath", yamlidx=0, name="name")
        with pytest.raises(TypeError):
            common.StructParseBaseNamed("yamlpath", 0, "name")
    else:
        assert common.KW_ONLY_DATACLASS == {}
        common.StructParseBaseNamed("yamlpath", yamlidx=0, name="name")
        common.StructParseBaseNamed("yamlpath", 0, "name")


def test_etc_hosts_update() -> None:
    assert common.etc_hosts_update_data("", {}) == ""
    assert common.etc_hosts_update_data("a", {}) == "a\n"

    assert (
        common.etc_hosts_update_data(
            "",
            {
                "foo": ("192.168.1.3", []),
            },
        )
        == "192.168.1.3 foo\n"
    )
    assert (
        common.etc_hosts_update_data(
            "10.2.3.4 foo\n",
            {
                "foo": ("192.168.1.3", None),
            },
        )
        == "192.168.1.3 foo\n"
    )
    assert (
        common.etc_hosts_update_data(
            "  \t 10.2.3.4\tfoo foo.alias\t",
            {
                "foo": ("192.168.1.3", ["foo2"]),
                "bar": ("192.168.1.1", None),
            },
        )
        == "192.168.1.3 foo foo2\n\n192.168.1.1 bar\n"
    )

    assert (
        common.etc_hosts_update_data(
            b"  1.1.1.1 xx\xcaxx\n  \t 10.2.3.4\tfoo foo.alias\t".decode(
                errors="surrogateescape"
            ),
            {
                "foo": ("192.168.1.3", ["foo2"]),
                "bar": ("192.168.1.1", None),
            },
        ).encode(errors="surrogateescape")
        == b"  1.1.1.1 xx\xcaxx\n192.168.1.3 foo foo2\n\n192.168.1.1 bar\n"
    )

    assert (
        common.etc_hosts_update_data(
            """127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6

172.131.100.100 marvell-dpu-42 dpu
""",
            {
                "marvell-dpu-42": ("172.131.100.100", ["dpu"]),
            },
        )
        == """127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6

172.131.100.100 marvell-dpu-42 dpu
"""
    )

    assert (
        common.etc_hosts_update_data(
            """127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6

172.131.100.100 marvell-dpu-42 dpu
""",
            {
                "marvell-dpu-42": ("172.131.100.100", ["dpu2"]),
            },
        )
        == """127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6

172.131.100.100 marvell-dpu-42 dpu2
"""
    )


def test_serial() -> None:
    try:
        import serial
    except ModuleNotFoundError:
        pytest.skip("pyserial module not available")

    with pytest.raises(serial.serialutil.SerialException):
        common.Serial("")


def _pargs(
    arg: Union[common._MISSING_TYPE, Optional[Any]] = common.MISSING,
    *,
    vdict: Optional[dict[str, Any]] = None,
    key: str = "foo",
) -> StructParsePopContext:
    if vdict is None:
        vdict = {}
    if not isinstance(arg, common._MISSING_TYPE):
        vdict[key] = arg
    return StructParsePopContext(vdict, "path", key)


def test_structparse_with() -> None:
    with common.structparse_with_strdict({}, "path") as pargs:
        with pytest.raises(ValueError):
            v1 = common.structparse_pop_str(pargs.for_key("v1"))
            if sys.version_info >= (3, 11):
                typing.assert_type(v1, str)

        v22 = common.structparse_pop_list(_pargs())
        if sys.version_info >= (3, 11):
            typing.assert_type(v22, list[Any])
        assert v22 == []

        v33n = common.structparse_pop_list(pargs.for_key("v33n"))
        if sys.version_info >= (3, 11):
            typing.assert_type(v33n, list[Any])
        assert v33n == []

        v3 = common.structparse_pop_enum(
            pargs.for_key("v3"),
            enum_type=TstTestType,
            default=TstTestType.IPERF_TCP,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(v3, TstTestType)
        assert v3 == TstTestType.IPERF_TCP

        v4 = common.structparse_pop_enum(
            pargs.for_key("v3"),
            enum_type=TstTestType,
            default=None,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(v4, Optional[TstTestType])
        assert v4 is None

    pctx = StructParseParseContext({"foo": "a"}, yamlpath="path")
    with pctx.with_strdict() as pargs:
        assert common.structparse_pop_str(pargs.for_key("foo")) == "a"


def test_structparse_pop_str_1() -> None:
    with pytest.raises(ValueError):
        val0 = common.structparse_pop_str(_pargs())
        if sys.version_info >= (3, 11):
            typing.assert_type(val0, str)

    val1 = common.structparse_pop_str(
        _pargs(),
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val1, Optional[str])
    assert val1 is None

    val2 = common.structparse_pop_str(
        _pargs(),
        default="defval",
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val2, str)
    assert val2 == "defval"

    val3 = common.structparse_pop_str(
        _pargs("strval"),
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val3, str)
    assert val3 == "strval"

    val4 = common.structparse_pop_str(
        _pargs("strval"),
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val4, Optional[str])
    assert val4 == "strval"

    val5 = common.structparse_pop_str(
        _pargs("strval"),
        default="defval",
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val5, str)
    assert val5 == "strval"

    val6 = common.structparse_pop_str(
        _pargs("strval"),
        check=lambda val: True,
    )
    assert val6 == "strval"

    with pytest.raises(ValueError):
        common.structparse_pop_str(
            _pargs("strval"),
            check=lambda val: False,
        )

    assert (
        common.structparse_pop_str(
            _pargs("strval"),
            check_ctx=lambda pargs2, val: None,
        )
        == "strval"
    )

    def _check_none() -> None:
        return None

    assert (
        common.structparse_pop_str(
            _pargs("strval"),
            check_ctx=lambda pargs2, val: _check_none(),
        )
        == "strval"
    )

    def _check1(pargs: StructParsePopContext, val: str) -> None:
        assert pargs.key == "foo"
        assert val == "strval"

    common.structparse_pop_str(
        _pargs("strval"),
        check_ctx=lambda pargs2, val: _check1(pargs2, val),
    )

    val8 = common.structparse_pop_str(
        _pargs("strval"),
        check_ctx=_check1,
    )
    assert val8 == "strval"

    def _check2(pargs: StructParsePopContext, val: str) -> None:
        assert pargs.key == "foo"
        assert val == "strval"
        raise pargs.value_error("hi")

    with pytest.raises(ValueError):
        common.structparse_pop_str(
            _pargs("strval"),
            check_ctx=_check2,
        )

    if sys.version_info >= (3, 11):
        common.structparse_pop_str(
            _pargs("strval"),
            check=lambda val: bool(typing.assert_type(val, str)) or True,
        )

    if sys.version_info >= (3, 11):
        common.structparse_pop_str(
            _pargs("strval"),
            check_ctx=lambda pargs2, val: (
                None if typing.assert_type(val, str) else None
            ),
        )


def test_structparse_pop_str_empty() -> None:
    def _check(x: str) -> bool:
        assert x == ""
        raise ReachedError("I am here")

    def rnd_ntf() -> Optional[bool]:
        return random.choice([None, True, False])

    def rnd_check() -> Optional[typing.Callable[[str], bool]]:
        return random.choice([None, _check])

    def rnd_default() -> Optional[str]:
        return random.choice([None, "xxx"])

    # if nothing specified, empty values are rejected.
    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""))

    # If we have a default, the empty value maps to the default.
    assert common.structparse_pop_str(_pargs(""), default=None) is None
    assert common.structparse_pop_str(_pargs(""), default="xxx") == "xxx"

    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), empty_as_default=None)
    assert common.structparse_pop_str(_pargs(""), empty_as_default=False) == ""
    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), empty_as_default=True)

    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), allow_empty=None)
    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), allow_empty=False)
    assert common.structparse_pop_str(_pargs(""), allow_empty=True) == ""

    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), check=None)
    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), check=_check)

    for i in range(1000):

        with pytest.raises(ValueError):
            common.structparse_pop_str(
                _pargs(""),
                default=random.choice([common.MISSING, None, "xxx"]),
                empty_as_default=random.choice([None, False, True]),
                allow_empty=False,
                check=random.choice([None, _check]),
            )

        d1 = random.choice([None, "xxx"])
        assert (
            common.structparse_pop_str(
                _pargs(""),
                default=d1,
                empty_as_default=True,
                allow_empty=random.choice([None, True]),
                check=random.choice([None, _check]),
            )
            is d1
        )

        assert (
            common.structparse_pop_str(
                _pargs(""),
                default=random.choice([None, "xxx"]),
                empty_as_default=False,
                allow_empty=random.choice([None, True]),
                check=None,
            )
            == ""
        )
        with pytest.raises(ReachedError):
            common.structparse_pop_str(
                _pargs(""),
                default=random.choice([None, "xxx"]),
                empty_as_default=False,
                allow_empty=random.choice([None, True]),
                check=_check,
            )

    assert (
        common.structparse_pop_str(
            _pargs(""),
            default="xxx",
            empty_as_default=False,
        )
        == ""
    )

    assert common.structparse_pop_str(_pargs(""), default="xxx", check=_check) == "xxx"
    assert (
        common.structparse_pop_str(
            _pargs(""), default="xxx", empty_as_default=True, check=_check
        )
        == "xxx"
    )
    with pytest.raises(ReachedError):
        common.structparse_pop_str(
            _pargs(""), default="xxx", empty_as_default=False, check=_check
        )

    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), empty_as_default=True, check=_check)
    with pytest.raises(ReachedError):
        common.structparse_pop_str(_pargs(""), empty_as_default=False, check=_check)

    with pytest.raises(ReachedError):
        common.structparse_pop_str(_pargs(""), allow_empty=True, check=_check)
    with pytest.raises(ValueError):
        common.structparse_pop_str(_pargs(""), allow_empty=False, check=_check)

    with pytest.raises(ValueError):
        common.structparse_pop_str_name(_pargs())

    with pytest.raises(ValueError):
        common.structparse_pop_str_name(_pargs(""))

    with pytest.raises(ValueError):
        common.structparse_pop_str_name(
            _pargs(""),
            check_ctx=lambda pargs, val: None,
        )

    def _check1(pargs: StructParsePopContext, val: str) -> None:
        assert pargs.key == "foo"
        assert val == "strval"

    assert (
        common.structparse_pop_str_name(
            _pargs("strval"),
            check_ctx=_check1,
        )
        == "strval"
    )

    with pytest.raises(AssertionError):
        common.structparse_pop_str(
            _pargs("invalid"),
            check_ctx=_check1,
        )

    assert (
        common.structparse_pop_str_name(
            _pargs("hi"),
            check_regex="i",
        )
        == "hi"
    )

    assert (
        common.structparse_pop_str(
            _pargs("hi"),
            check=lambda val: True,
            check_regex="h",
        )
        == "hi"
    )

    with pytest.raises(ValueError):
        common.structparse_pop_str(
            _pargs("hix"),
            check_regex=re.compile("^hi$"),
        )

    with pytest.raises(ValueError):
        common.structparse_pop_str(
            _pargs("hix"),
            check_regex="xx",
        )


def test_structparse_pop_bool() -> None:
    for test_val in (None, "", "default", "-1", "bogus"):
        with pytest.raises(ValueError):
            val0c = common.structparse_pop_bool(
                _pargs(test_val),
            )
            if sys.version_info >= (3, 11):
                typing.assert_type(val0c, bool)

    with pytest.raises(ValueError):
        val0a = common.structparse_pop_bool(
            _pargs("bogus"),
            default=False,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0a, bool)

    with pytest.raises(ValueError):
        val0b = common.structparse_pop_bool(
            _pargs("bogus"),
            default=None,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0b, Optional[bool])

    for test_val in (None, "", "default", "-1"):
        val0d = common.structparse_pop_bool(
            _pargs(test_val),
            default=False,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0d, bool)
        assert val0d is False

    for test_val in (None, "", "default", "-1"):
        val0e = common.structparse_pop_bool(
            _pargs(test_val),
            default=None,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0e, Optional[bool])
        assert val0e is None

    val1 = common.structparse_pop_bool(
        _pargs(),
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val1, Optional[bool])
    assert val1 is None

    val2 = common.structparse_pop_bool(
        _pargs(),
        default=False,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val2, bool)
    assert val2 is False

    val3 = common.structparse_pop_bool(
        _pargs("true"),
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val3, bool)
    assert val3 is True

    val4 = common.structparse_pop_bool(
        _pargs("1"),
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val4, Optional[bool])
    assert val4 is True

    val5 = common.structparse_pop_bool(
        _pargs("yes"),
        default=False,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val5, bool)
    assert val5 is True

    assert (
        common.structparse_pop_bool(
            _pargs(),
            default=None,
        )
        is None
    )

    assert (
        common.structparse_pop_bool(
            _pargs(),
            default=None,
            check_ctx=lambda pargs, val: common.raise_exception(RuntimeError()),
        )
        is None
    )

    with pytest.raises(ValueError):
        common.structparse_pop_bool(
            _pargs(1),
            check_ctx=lambda pargs, val: common.raise_exception(RuntimeError()),
        )

    with pytest.raises(RuntimeError):
        common.structparse_pop_bool(
            _pargs(True),
            check_ctx=lambda pargs, val: common.raise_exception(RuntimeError()),
        )


def test_structparse_pop_enum() -> None:
    with pytest.raises(ValueError):
        val0 = common.structparse_pop_enum(
            _pargs(),
            enum_type=TstTestType,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0, TstTestType)

    val1 = common.structparse_pop_enum(
        _pargs(),
        enum_type=TstTestType,
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val1, Optional[TstTestType])
    assert val1 is None

    val2 = common.structparse_pop_enum(
        _pargs(),
        enum_type=TstTestType,
        default=TstTestType.IPERF_TCP,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val2, TstTestType)
    assert val2 == TstTestType.IPERF_TCP

    val3 = common.structparse_pop_enum(
        _pargs("http"),
        enum_type=TstTestType,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val3, TstTestType)
    assert val3 == TstTestType.HTTP

    val4 = common.structparse_pop_enum(
        _pargs("http"),
        enum_type=TstTestType,
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val4, Optional[TstTestType])
    assert val4 == TstTestType.HTTP

    val5 = common.structparse_pop_enum(
        _pargs("http"),
        enum_type=TstTestType,
        default=TstTestType.IPERF_TCP,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val5, TstTestType)
    assert val5 == TstTestType.HTTP


def test_structparse_pop_obj() -> None:
    def _construct(pctx: StructParseParseContext) -> int:
        if pctx.arg is None:
            return -1
        return int(pctx.arg)

    v1 = common.structparse_pop_obj(
        _pargs("4"),
        construct=_construct,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(v1, int)
    assert v1 == 4

    with pytest.raises(ValueError):
        v2 = common.structparse_pop_obj(
            _pargs(None),
            construct=_construct,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(v2, int)
        assert False

    v3 = common.structparse_pop_obj(
        _pargs(None),
        construct=_construct,
        default=None,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(v3, Optional[int])
    assert v3 is None

    v4 = common.structparse_pop_obj(
        _pargs(None),
        construct=_construct,
        default=99,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(v4, int)
    assert v4 == 99

    v5 = common.structparse_pop_obj(
        _pargs(None),
        construct=_construct,
        default="99",
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(v5, Union[int, str])
    assert v5 == "99"

    v6 = common.structparse_pop_obj(
        _pargs(None),
        construct=_construct,
        construct_default=True,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(v6, int)
    assert v6 == -1

    assert (
        common.structparse_pop_obj(
            _pargs(),
            default=None,
            construct=lambda pctx: common.raise_exception(RuntimeError()),
            check_ctx=lambda pargs, obj: common.raise_exception(RuntimeError()),
        )
        is None
    )

    assert (
        common.structparse_pop_obj(
            _pargs("xxx"),
            construct=lambda pctx: pctx.arg,
            check_ctx=lambda pargs, obj: obj.strip(),
        )
        == "xxx"
    )

    assert (
        common.structparse_pop_obj(
            _pargs("xxx"),
            construct=lambda pctx: typing.cast(str, pctx.arg),
            check_ctx=lambda pargs, obj: None if obj.strip() else None,
        )
        == "xxx"
    )


def test_structparse_pop_list() -> None:
    lst1 = common.structparse_pop_list(_pargs())
    if sys.version_info >= (3, 11):
        typing.assert_type(lst1, list[Any])
    assert lst1 == []

    lst2 = common.structparse_pop_list(_pargs())
    if sys.version_info >= (3, 11):
        typing.assert_type(lst2, list[Any])
    assert lst2 == []

    with pytest.raises(ValueError):
        lst7 = common.structparse_pop_list(_pargs(), allow_missing=False)
        if sys.version_info >= (3, 11):
            typing.assert_type(lst7, list[Any])
        assert False

    lst9 = common.structparse_pop_list(_pargs(), allow_missing=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(lst9, list[Any])
    assert lst9 == []

    with pytest.raises(ValueError):
        lst17 = common.structparse_pop_list(_pargs(), allow_empty=False)
        if sys.version_info >= (3, 11):
            typing.assert_type(lst17, list[Any])
        assert False

    lst19 = common.structparse_pop_list(_pargs(), allow_empty=True)
    if sys.version_info >= (3, 11):
        typing.assert_type(lst19, list[Any])
    assert lst19 == []

    lst8 = common.structparse_pop_list(_pargs([]), allow_missing=False)
    if sys.version_info >= (3, 11):
        typing.assert_type(lst8, list[Any])
    assert lst8 == []

    lst21 = common.structparse_pop_list(
        _pargs([]),
        allow_missing=False,
        allow_empty=True,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(lst21, list[Any])
    assert lst21 == []


def test_structparse_pop_objlist_as_dict() -> None:
    def do_dict(
        key: str,
        vdict: dict[str, Any],
        *,
        allow_duplicates: bool = False,
    ) -> dict[int, tuple[int, str, int]]:
        val0 = common.structparse_pop_objlist_to_dict(
            _pargs(vdict=vdict, key=key),
            construct=lambda pctx: (pctx.yamlidx, pctx.yamlpath, int(pctx.arg)),
            get_key=lambda v: int(v[2]),
            allow_duplicates=allow_duplicates,
        )
        if sys.version_info >= (3, 11):
            typing.assert_type(val0, dict[int, tuple[int, str, int]])
        return val0

    def do(
        vlst: list[Any],
        *,
        allow_duplicates: bool = False,
    ) -> dict[int, tuple[int, str, int]]:
        return do_dict("foo", {"foo": vlst}, allow_duplicates=allow_duplicates)

    assert do_dict("foo", {}) == {}
    assert do_dict("foo", {"foo": []}) == {}
    assert do([]) == {}
    val7: dict[int, tuple[int, str, int]] = do(["1"])
    assert val7 == {1: (0, "path.foo[0]", 1)}
    assert do(["1", "2"]) == {
        1: (0, "path.foo[0]", 1),
        2: (1, "path.foo[1]", 2),
    }
    with pytest.raises(ValueError) as ex:
        do(["1", "2", "1"])
    assert str(ex.value) == '"path.foo[2]": duplicate key \'1\' with "path.foo[0]"'
    assert do(["1", "2", "1"], allow_duplicates=True) == {
        2: (1, "path.foo[1]", 2),
        1: (2, "path.foo[2]", 1),
    }

    val1 = common.structparse_pop_objlist_to_dict(
        _pargs([0, 1, 2]),
        construct=lambda pctx: common.StructParseBaseNamed(
            yamlpath=pctx.yamlpath,
            yamlidx=pctx.yamlidx,
            name=str(pctx.arg),
        ),
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val1, dict[str, common.StructParseBaseNamed])
    assert val1 == {
        "0": common.StructParseBaseNamed(yamlidx=0, yamlpath="path.foo[0]", name="0"),
        "1": common.StructParseBaseNamed(yamlidx=1, yamlpath="path.foo[1]", name="1"),
        "2": common.StructParseBaseNamed(yamlidx=2, yamlpath="path.foo[2]", name="2"),
    }

    val2 = common.structparse_pop_objlist_to_dict(
        _pargs([0, 1, 2]),
        construct=lambda pctx: str(pctx.arg),
        get_key=lambda x: x,
    )
    if sys.version_info >= (3, 11):
        typing.assert_type(val2, dict[str, str])

    with pytest.raises(RuntimeError):
        common.structparse_pop_objlist_to_dict(  # type: ignore
            _pargs([0, 1, 2]),
            construct=lambda pctx: str(pctx.arg),
        )


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


def test_path_norm() -> None:

    @dataclasses.dataclass
    class Conf:
        arg: str
        result: str
        normpath: Optional[str] = None
        cwd: Optional[str] = None
        preserve_dir: bool = True

        @property
        def cwd_randomly_as_path(self) -> Optional[Union[str, pathlib.Path]]:
            if self.cwd is None:
                return None
            return random.choice([conf.cwd, pathlib.Path(self.cwd)])

        @property
        def result_no_trailing_slash(self) -> str:
            if len(self.result) > 1 and self.result[-1] == "/":
                return self.result[:-1]
            return self.result

    confs = [
        Conf("", "."),
        Conf("", "foo/", cwd="foo", normpath="."),
        Conf("", "foo/", cwd="foo/", normpath="."),
        Conf("", "foo", cwd="foo", preserve_dir=False, normpath="."),
        Conf("", "foo", cwd="./foo/", preserve_dir=False, normpath="."),
        Conf(".", "."),
        Conf("/tmp/.", "/tmp/", normpath="/tmp"),
        Conf("/a/..//././bbb///", "/a/../bbb/", normpath="/bbb"),
        Conf("//..//././bbb///", "/../bbb/", normpath="//bbb"),
        Conf("/a/..//././bbb/..///", "/a/../bbb/../", normpath="/"),
        Conf("/tmp", "/tmp"),
        Conf("tmp", "tmp"),
        Conf("tmp", "/tmp", normpath="tmp", cwd="/"),
        Conf("/tmp/", "/tmp/", normpath="/tmp"),
        Conf("/tmp/.", "/tmp/", normpath="/tmp"),
        Conf("/tmp/.", "/tmp", preserve_dir=False),
        Conf("/tmp/", "/tmp", preserve_dir=False),
        Conf("/tmp/..", "/tmp/..", normpath="/", preserve_dir=False),
        Conf("/tmp/..", "/tmp/../", normpath="/", preserve_dir=True),
    ]
    for conf in confs:
        r1 = common.path_norm(
            conf.arg,
            cwd=conf.cwd_randomly_as_path,
            preserve_dir=conf.preserve_dir,
        )
        assert isinstance(r1, str)
        assert r1 == conf.result
        if conf.arg == conf.result:
            assert r1 is conf.arg

        parg = pathlib.Path(conf.arg)
        p1 = common.path_norm(
            parg,
            cwd=conf.cwd_randomly_as_path,
            preserve_dir=conf.preserve_dir,
        )
        assert isinstance(p1, pathlib.Path)
        assert str(p1) == conf.result_no_trailing_slash
        if conf.arg == conf.result:
            assert p1 is parg

        r2 = os.path.normpath(conf.arg)
        assert isinstance(r1, str)
        if conf.normpath is None:
            assert r2 == conf.result
        else:
            assert r2 == conf.normpath
            assert conf.result != conf.normpath

    assert common.path_norm("foo", cwd="path") == "path/foo"
    assert common.path_norm("foo", cwd="path") == "path/foo"

    s1 = common.path_norm("foo", cwd="path", make_absolute=True)
    assert s1 == os.path.join(os.getcwd(), "path/foo")
    assert os.path.isabs(s1)

    s1 = common.path_norm("foo", cwd="/path", make_absolute=True)
    assert s1 == "/path/foo"
    assert os.path.isabs(s1)


def test_iter_listify() -> None:
    @common.iter_listify
    def f_lst_1() -> Iterable[int]:
        yield 1
        yield 2

    if sys.version_info >= (3, 11):
        typing.assert_type(f_lst_1(), list[int])

    x_lst_1: list[int] = f_lst_1()
    assert x_lst_1 == [1, 2]

    @common.iter_listify
    def f_lst_2() -> tuple[str, ...]:
        return ("a", "b")

    if sys.version_info >= (3, 11):
        typing.assert_type(f_lst_2(), list[str])

    x_lst_2: list[str] = f_lst_2()
    assert x_lst_2 == ["a", "b"]

    @common.iter_listify
    def f_lst_3(*args: tuple[str, int]) -> dict[str, int]:
        return dict(args)

    if sys.version_info >= (3, 11):
        typing.assert_type(f_lst_3(), list[str])

    x_lst_3: list[str] = f_lst_3()
    assert x_lst_3 == []

    x_lst_3 = f_lst_3(("a", 1), ("b", 2))
    assert x_lst_3 == ["a", "b"]

    @common.iter_tuplify
    def f_tpl_1() -> Iterable[str]:
        yield "1"

    if sys.version_info >= (3, 11):
        typing.assert_type(f_tpl_1(), tuple[str, ...])

    x_tpl_1: tuple[str, ...] = f_tpl_1()
    assert x_tpl_1 == ("1",)

    @common.iter_tuplify
    @common.iter_tuplify
    def f_tpl_2(x: float) -> list[float]:
        return [1.4, 2.0, x]

    if sys.version_info >= (3, 11):
        typing.assert_type(f_tpl_2(5.0), tuple[float, ...])

    x_tpl_2: tuple[float, ...] = f_tpl_2(5.0)
    assert x_tpl_2 == (1.4, 2.0, 5.0)

    @common.iter_dictify
    def f_dict_1(a: int, b: str, c: str) -> Iterable[tuple[int, str]]:
        return {a: b + c}.items()

    if sys.version_info >= (3, 11):
        typing.assert_type(f_dict_1(2, "", "x"), dict[int, str])

    x_dict_1: dict[int, str] = f_dict_1(1, "b", "x")
    assert x_dict_1 == {1: "bx"}

    @common.iter_dictify
    @common.iter_tuplify
    def f_dict_2(a: int, b: str, c: str) -> Iterable[tuple[int, str]]:
        yield a, b
        yield 5, c

    if sys.version_info >= (3, 11):
        typing.assert_type(f_dict_2(2, "", "x"), dict[int, str])

    x_dict_2: dict[int, str] = f_dict_2(1, "b", "x")
    assert x_dict_2 == {1: "b", 5: "x"}

    f_dict_2_x: typing.Callable[[int, str, str], dict[int, str]] = f_dict_2
    assert f_dict_2_x is f_dict_2


def test_iter_eval_now() -> None:
    lst1 = common.iter_eval_now(x for x in range(5))
    assert lst1 == (0, 1, 2, 3, 4)
    assert len(lst1) == 5
    assert lst1[3] == 3


def test_json_dump() -> None:
    def _dump(data: Any) -> str:
        buffer = io.StringIO()
        common.json_dump(data, buffer)
        return buffer.getvalue()

    assert _dump(1) == "1\n"
    assert (
        _dump({"a": 5.5})
        == """{
  "a": 5.5
}
"""
    )


def test_file_or_open(tmp_path: pathlib.Path) -> None:
    tmp_file = tmp_path / "file1"

    with common.use_or_open(tmp_file, mode="w") as f:
        f.write("hello")

    with open(tmp_file) as f1:
        with common.use_or_open(f1) as f:
            assert f.read() == "hello"

    buffer = io.StringIO()
    with common.use_or_open(buffer, mode="w") as f:
        f.write("hello")
    assert buffer.getvalue() == "hello"


def test_argparse_regex_type() -> None:

    pattern = common.argparse_regex_type("regex")
    assert isinstance(pattern, re.Pattern)
    assert pattern.pattern == "regex"

    with pytest.raises(argparse.ArgumentTypeError):
        common.argparse_regex_type("broken1[")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=common.argparse_regex_type)

    args = parser.parse_args(["--pattern", "regex"])
    assert isinstance(args.pattern, re.Pattern)
    assert args.pattern.pattern == "regex"

    with pytest.raises(SystemExit) as ex:
        parser.parse_args(["--pattern", "broken["])
    assert isinstance(ex.value, SystemExit)


def test_thread_list() -> None:
    lst = (host.local.run_in_thread("echo"),)
    common.thread_list_cancel(threads=lst)
    common.thread_list_join_all(threads=lst, cancel=True)
    common.thread_list_join_all(threads=lst)

    lst2: tuple[threading.Thread, ...] = ()
    common.thread_list_cancel(threads=lst2)
    common.thread_list_join_all(threads=lst2)


def test_format_duration() -> None:
    assert common.format_duration(125.5) == "00:02:05.5000"
    assert common.format_duration(125) == "00:02:05.0000"
    assert common.format_duration(0) == "00:00:00.0000"
    assert common.format_duration(-0.5) == "-00:00:00.5000"
    assert common.format_duration(5) == "00:00:05.0000"
    assert common.format_duration(5.25) == "00:00:05.2500"
    assert common.format_duration(60) == "00:01:00.0000"
    assert common.format_duration(60.75) == "00:01:00.7500"
    assert common.format_duration(3599) == "00:59:59.0000"
    assert common.format_duration(3599.999) == "00:59:59.9990"
    assert common.format_duration(3600) == "01:00:00.0000"
    assert common.format_duration(3600.5) == "01:00:00.5000"
    assert common.format_duration(3661.125) == "01:01:01.1250"
    assert common.format_duration(7325.75) == "02:02:05.7500"
    assert common.format_duration(-45) == "-00:00:45.0000"
    assert common.format_duration(-3661.8) == "-01:01:01.8000"
    assert common.format_duration(36000) == "10:00:00.0000"
    assert common.format_duration(86399.9999) == "23:59:59.9999"
    assert common.format_duration(86400) == "1d-00:00:00.0000"
    assert common.format_duration(-86400) == "-1d-00:00:00.0000"


def test_env_get_ktoolbox_logfile_parse() -> None:
    parse = common._env_get_ktoolbox_logfile_parse

    rep_p = str(os.getpid())
    rep_h = socket.gethostname().split(".", 1)[0]
    rep_t = str(common.get_program_epoch())
    rep_a = common.get_program_appname(with_pysuffix=False)

    assert parse("") is None
    assert parse("x") == ("x", None, True)
    assert parse("::x") == (":x", None, True)
    assert parse("   ::x") == (":x", None, True)
    assert parse("+x") == ("x", None, True)
    assert parse("=x") == ("x", None, False)
    assert parse("x:=x") == ("x:=x", None, True)
    assert parse("  debuG :=x") == ("x", logging.DEBUG, False)
    assert parse(":+x") == ("x", None, True)
    assert parse("debug:x") == ("x", logging.DEBUG, True)
    assert parse("debug:=x") == ("x", logging.DEBUG, False)
    assert parse("DEBUG:+x") == ("x", logging.DEBUG, True)
    assert parse("DEBUG:+x%p%h") == (f"x{rep_p}{rep_h}", logging.DEBUG, True)
    assert parse("DEBUG:+x%%p%h") == (f"x%p{rep_h}", logging.DEBUG, True)
    assert parse("invalid_level:x") == ("invalid_level:x", None, True)
    assert parse("file.txt") == ("file.txt", None, True)
    assert parse("%%") == ("%", None, True)
    assert parse("%p") == (rep_p, None, True)
    assert parse("%h") == (rep_h, None, True)
    assert parse("test_%p_%h.log") == (f"test_{rep_p}_{rep_h}.log", None, True)
    assert parse("file_%%p%%h.log") == ("file_%p%h.log", None, True)
    assert parse("debug:+logfile.txt") == ("logfile.txt", logging.DEBUG, True)
    assert parse("info:=logfile.txt") == ("logfile.txt", logging.INFO, False)
    assert parse("WARNING:logfile.txt") == ("logfile.txt", logging.WARNING, True)
    assert parse("debug:+logfile.txt") == ("logfile.txt", logging.DEBUG, True)
    assert parse("info:+log_%p_%h.log") == (
        f"log_{rep_p}_{rep_h}.log",
        logging.INFO,
        True,
    )
    assert parse(":logfile.txt") == ("logfile.txt", None, True)
    assert parse("debug:foo:bar") == ("foo:bar", logging.DEBUG, True)
    assert parse("DeBuG:+logfile.txt") == ("logfile.txt", logging.DEBUG, True)
    assert parse("InFo:+logfile.txt") == ("logfile.txt", logging.INFO, True)
    assert parse("%%p%%h:.log") == ("%p%h:.log", None, True)
    assert parse("debug:foo:bar") == ("foo:bar", logging.DEBUG, True)
    assert parse("debug:+file_%%p%%h:.log") == ("file_%p%h:.log", logging.DEBUG, True)
    assert parse("debug:") is None
    assert parse("+") is None
    assert parse("=") is None
    assert parse(" info :+") is None
    assert parse(" info :x") == ("x", logging.INFO, True)
    assert parse("file_%%p%h.log") == (f"file_%p{rep_h}.log", None, True)
    assert parse("file_%%p%h%p%h%p.log") == (
        f"file_%p{rep_h}{rep_p}{rep_h}{rep_p}.log",
        None,
        True,
    )
    assert parse("debug:logfile-%a.txt") == (
        f"logfile-{rep_a}.txt",
        logging.DEBUG,
        True,
    )
    assert parse("debug:logfile-%t.txt") == (
        f"logfile-{rep_t}.txt",
        logging.DEBUG,
        True,
    )


def test_next_expiry() -> None:
    now = common.time_monotonic(None)

    next_expiry = common.NextExpiry()

    next_expiry.update(timeout=5)

    assert common.unwrap(next_expiry.expires_in()) <= 5.0
    assert common.unwrap(next_expiry.expires_at()) > now + 5.0

    next_expiry.reset()

    common.NextExpiry.up(next_expiry, now=now, timeout=7)
    assert next_expiry.expires_at() == now + 7.0


def test_validate_dns_name() -> None:
    assert common.validate_dns_name("example.com") is True
    assert common.validate_dns_name("sub.domain.example") is True
    assert common.validate_dns_name("a" * 63 + ".com") is True
    assert common.validate_dns_name("example.com.") is True
    assert common.validate_dns_name("xn--d1acpjx3f.xn--p1ai") is True

    assert common.validate_dns_name("") is False
    assert common.validate_dns_name(".") is False
    assert common.validate_dns_name("example..com") is False
    assert common.validate_dns_name("-example.com") is False
    assert common.validate_dns_name("example-.com") is False
    assert common.validate_dns_name("a" * 64 + ".com") is False
    assert common.validate_dns_name("example.com..") is False
    assert common.validate_dns_name("exa mple.com") is False
    assert common.validate_dns_name("example.com-") is False
    assert common.validate_dns_name("." * 254) is False

    with pytest.raises(ValueError):
        common.validate_dns_name(None)  # type: ignore

    with pytest.raises(ValueError):
        common.validate_dns_name(123)  # type: ignore


def test_sed_escape_repl() -> None:
    assert common.sed_escape_repl("") == ""
    assert common.sed_escape_repl("abc\\d/x&dd") == "abc\\\\d\\/x\\&dd"
