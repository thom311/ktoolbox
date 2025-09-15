import abc
import concurrent.futures
import contextlib
import dataclasses
import functools
import json
import logging
import os
import pathlib
import re
import socket
import sys
import threading
import time
import typing
import weakref

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeVar
from typing import Union

if typing.TYPE_CHECKING:
    # https://github.com/python/typeshed/tree/main/stdlib/_typeshed#api-stability
    # https://github.com/python/typeshed/blob/6220c20d9360b12e2287511587825217eec3e5b5/stdlib/_typeshed/__init__.pyi#L349
    from _typeshed import DataclassInstance
    from types import TracebackType
    import argparse


FATAL_EXIT_CODE = 255


common_lock = threading.Lock()


PathType = Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]

E = TypeVar("E", bound=Enum)
T = TypeVar("T")
TOptionalStr = TypeVar("TOptionalStr", bound=Optional[str])
TOptionalInt = TypeVar("TOptionalInt", bound=Optional[int])
TOptionalBool = TypeVar("TOptionalBool", bound=Optional[bool])
TOptionalFloat = TypeVar("TOptionalFloat", bound=Optional[float])
T1 = TypeVar("T1")
T2 = TypeVar("T2")
TAnyStr = TypeVar("TAnyStr", str, bytes)
TCallable = typing.TypeVar("TCallable", bound=typing.Callable[..., typing.Any])
TStructParseBaseNamed = typing.TypeVar(
    "TStructParseBaseNamed", bound="StructParseBaseNamed"
)


# This is used as default value for some arguments, to recognize that the
# caller didn't specify the argument. This is useful, when we want to
# explicitly distinguish between having an argument unset or set to any value.
# The caller would never pass this value, but the implementation would check
# whether the argument is still left at the default.
#
# See also, dataclasses.MISSING and dataclasses._MISSING_TYPE
class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()

# kw_only is Python3.10+. This annotation is very useful, so make it available
# with 3.9 without breaking mypy.
#
# This silences up mypy while retaining the error checking at runtime. It however
# looses the ability for mypy to detect the error at lint time.
#
# So, using this acts as a comment to the reader that the code expects kw_only.
# It is also enforced at runtime. It also allows to find all the places where
# we would like to use kw_only=True but cannot due to Python 3.9 compatibility.
KW_ONLY_DATACLASS = (
    {"kw_only": True}
    if (dataclass.__kwdefaults__ and "kw_only" in dataclass.__kwdefaults__)
    else {}
)


def raise_exception(ex: Exception) -> typing.NoReturn:
    # raises exception. This exists because raise is a statement that cannot be
    # used at certain places (like a lambda function).
    raise ex


def char_is_surrogateescaped(char: str) -> bool:
    """Returns True if the character is a surrogate escape (i.e., originally an invalid UTF-8 byte)."""
    return 0xDC80 <= ord(char) <= 0xDCFF


BOOL_TO_STR_FORMATS: Mapping[str, tuple[str, str]] = {
    "true": ("true", "false"),
    "yes": ("yes", "no"),
    "1": ("1", "0"),
    "on": ("on", "off"),
}


def bool_to_str(val: bool, *, format: str = "true") -> str:
    try:
        fmt_tuple = BOOL_TO_STR_FORMATS[format]
    except KeyError:
        raise ValueError(f'Invalid format "{format}"')
    return fmt_tuple[0] if val else fmt_tuple[1]


def str_to_bool(
    val: Optional[Union[str, bool]],
    on_error: Union[T1, _MISSING_TYPE] = MISSING,
    *,
    on_default: Union[T2, _MISSING_TYPE] = MISSING,
) -> Union[bool, T1, T2]:

    is_default = False

    if isinstance(val, str):
        val2 = val.lower().strip()
        if val2 in ("1", "y", "yes", "true", "on"):
            return True
        if val2 in ("0", "n", "no", "false", "off"):
            return False
        if val2 in ("", "default", "-1"):
            is_default = True
    elif val is None:
        # None is (maybe) accepted as default value.
        is_default = True
    elif isinstance(val, bool):
        # For convenience, also accept that the value is already a boolean.
        return val

    if is_default and not isinstance(on_default, _MISSING_TYPE):
        # The value is explicitly set to one of the recognized default values
        # (None, "default", "-1" or "").
        #
        # By setting @on_default, the caller can use str_to_bool() to not only
        # parse boolean values, but ternary values.
        return on_default

    if not isinstance(on_error, _MISSING_TYPE):
        # On failure, we return the fallback value.
        return on_error

    raise ValueError(f"Value {val} is not a boolean")


@typing.overload
def as_regex(
    pattern: Union[TAnyStr, re.Pattern[TAnyStr]],
    *,
    flags: int = 0,
) -> re.Pattern[TAnyStr]: ...


@typing.overload
def as_regex(
    pattern: Optional[Union[TAnyStr, re.Pattern[TAnyStr]]],
    *,
    flags: int = 0,
) -> Optional[re.Pattern[TAnyStr]]: ...


def as_regex(
    pattern: Optional[Union[TAnyStr, re.Pattern[TAnyStr]]],
    *,
    flags: int = 0,
) -> Optional[re.Pattern[TAnyStr]]:
    if pattern is None:
        return None
    if isinstance(pattern, re.Pattern):
        return pattern
    if isinstance(pattern, (str, bytes)):
        return re.compile(pattern, flags)
    raise ValueError("not a valid regex pattern")


def sed_escape_repl(s: str, *, delimiter: str = "/") -> str:
    # This is to escape strings in sed argument like
    # f"s/pattern/{replacement}/".
    s = s.replace("\\", "\\\\")
    s = s.replace("&", "\\&")
    if delimiter:
        if len(delimiter) != 1:
            raise ValueError("sed_escape_repl() requires a delimiter of length 1")
        s = s.replace(delimiter, "\\" + delimiter)
    return s


def validate_dns_name(name: str) -> bool:
    if not isinstance(name, str):
        raise ValueError("dns name must be a string")
    if not name or len(name) > 253:
        return False
    if name[-1] == ".":
        # One trailing dot is accepted, it means an absolute name.
        name = name[:-1]
    # - (?!-) don't start label hyphen
    # - [A-Za-z0-9-]{1,63} contain 1 to 63 characters
    # - (?<!-) don't end with hyphen
    label_regex = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
    return all(label_regex.match(label) for label in name.split("."))


@typing.overload
def iter_get_first(
    lst: Iterable[T],
    *,
    unique: typing.Literal[True],
    force_unique: typing.Literal[True],
    single: bool = False,
) -> T: ...


@typing.overload
def iter_get_first(
    lst: Iterable[T],
    *,
    unique: bool = False,
    force_unique: bool = False,
    single: typing.Literal[True],
) -> T: ...


@typing.overload
def iter_get_first(
    lst: Iterable[T],
    *,
    unique: bool = False,
    force_unique: bool = False,
    single: bool = False,
) -> Optional[T]: ...


def iter_get_first(
    lst: Iterable[T],
    *,
    unique: bool = False,
    force_unique: bool = False,
    single: bool = False,
) -> Optional[T]:
    """
    Returns the first item from the iterable `lst` based on specified conditions.

    The function behaves differently depending on the parameters:

    - By default, if neither `unique`, `force_unique` or `single` is set, the
      function simply returns the first item from the iterable, or `None` if the
      iterable is empty.

    - If `unique=True`, it returns the first item if the iterable only contains
      a single element. Otherwise `None` is returned.

    - If `force_unique=True`, it ensures the iterable contains at most one
      item and raises a ValueError if multiple unique items are found. An
      empty iterable will give `None`.

    - Setting both `unique=True` and `force_unique=True` together or setting
      `single=True` enforces that the iterable contains exactly one element and
      returns it (raising an ValueError otherwise).

    Args:
        lst (Iterable[T]): The input iterable.
        unique (bool, optional): Returns `None` if the iterable contains multiple elements. Defaults to False.
        force_unique (bool, optional): Raises a ValueError if the iterable contains multiple elements. Defaults to False.
        single (bool, optional): Shorthand for `unique=True` and `force_unique=True` to raise a ValueError if the iterable does not contain exaclty one element. Defaults to False.

    Returns:
        Optional[T]: The first item from the iterable, or `None` if the iterable is empty. Raises an error if conditions set
        by `unique`, `force_unique`, or `single` are violated.
    """
    if single:
        # This is a shorthand for setting both "unique" and "force_unique".
        unique = True
        force_unique = True
    itr = iter(lst)
    try:
        v0 = next(itr)
    except StopIteration:
        if unique and force_unique:
            # Usually, an empty iterable is accepted, unless "unique" and
            # "force_unique" are both True.
            raise ValueError(
                "Iterable was expected to contain one element but was empty"
            )
        return None
    try:
        next(itr)
    except StopIteration:
        # There is only one element, we are good.
        pass
    else:
        # Handle multiple elements.
        if force_unique:
            raise ValueError("Iterable was expected to only contain one entry")
        if unique:
            return None
    return v0


def iter_filter_none(lst: Iterable[Optional[T]]) -> Iterable[T]:
    for v in lst:
        if v is not None:
            yield v


def iter_eval_now(lst: Iterable[T]) -> Sequence[T]:
    if isinstance(lst, (list, tuple)):
        return lst
    return tuple(lst)


if sys.version_info >= (3, 10):
    P = typing.ParamSpec("P")

if sys.version_info >= (3, 10):

    def iter_listify(
        fcn: typing.Callable[P, Iterable[T]],
    ) -> typing.Callable[P, list[T]]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> list[T]:
            return list(fcn(*args, **kwargs))

        return wrapper

else:

    def iter_listify(
        fcn: typing.Callable[..., Iterable[T]],
    ) -> typing.Callable[..., list[T]]:
        def wrapper(*args: Any, **kwargs: Any) -> list[T]:
            return list(fcn(*args, **kwargs))

        return wrapper


if sys.version_info >= (3, 10):

    def iter_tuplify(
        fcn: typing.Callable[P, Iterable[T]],
    ) -> typing.Callable[P, tuple[T, ...]]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, ...]:
            return tuple(fcn(*args, **kwargs))

        return wrapper

else:

    def iter_tuplify(
        fcn: typing.Callable[..., Iterable[T]],
    ) -> typing.Callable[..., tuple[T, ...]]:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[T, ...]:
            return tuple(fcn(*args, **kwargs))

        return wrapper


if sys.version_info >= (3, 10):

    def iter_dictify(
        fcn: typing.Callable[P, Iterable[tuple[T1, T2]]],
    ) -> typing.Callable[P, dict[T1, T2]]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[T1, T2]:
            return dict(fcn(*args, **kwargs))

        return wrapper

else:

    def iter_dictify(
        fcn: typing.Callable[..., Iterable[tuple[T1, T2]]],
    ) -> typing.Callable[..., dict[T1, T2]]:
        def wrapper(*args: Any, **kwargs: Any) -> dict[T1, T2]:
            return dict(fcn(*args, **kwargs))

        return wrapper


def unwrap(val: Optional[T], *, or_else: Optional[T] = None) -> T:
    # Like Rust's unwrap. Get the value or die (with an exception).
    #
    # The error message here is not good, so this function is more for
    # asserting (and shutting up the type checker) in cases where we
    # expect to have a value.
    if val is None:
        if or_else is not None:
            return or_else
        raise ValueError("Optional value unexpectedly not set")
    return val


TPathNormPath = TypeVar("TPathNormPath", bound=Union[None, str, pathlib.Path])


def path_norm(
    path: TPathNormPath,
    *,
    cwd: Optional[Union[str, pathlib.Path]] = None,
    preserve_dir: bool = False,
    make_absolute: bool = False,
) -> TPathNormPath:
    """
    Normalize a path while preserving symbolic links and other specific rules.

    Parameters:
    path (str): The path to normalize. None is allowed and results in None.
      pathlib.Path arguments are allowed and the result will also be a Path.
    cwd (Optional[str]): If provided, relative paths are joined with this base
      directory.
    preserve_dir (bool): If True and the path is a directory, the trailing
      slash is preserved.
    make_absolute: relative paths will always be completed with "cwd" (if provided).
      If "cwd" is not given or also a relative path, then the result is still a
      relative path. If `make_absolute` is given, such relative path is made
      absolute using os.getcwd().

    Returns:
    str: The normalized path.

    Notes:
    - This function is similar to `os.path.normpath()`, but with key differences:
      - Unlike `normpath()`, this function **does not remove `..`** components,
        preserving their meaning (important when symbolic links are involved).
      - `normpath()` keeps leading `//`, which is undesired in most cases. This
        function collapses all duplicate slashes into a single `/`.
    """
    if path is None:
        return typing.cast(TPathNormPath, None)

    path_orig = path

    if isinstance(path, pathlib.Path):
        path_str = str(path)
    else:
        assert isinstance(path, str)
        path_str = path

    path_orig_str = path_str

    is_abs = path_str.startswith("/")
    if not is_abs and cwd is not None:
        if isinstance(cwd, pathlib.Path):
            cwd = str(cwd)
        if cwd:
            path_str = os.path.join(cwd, path_str)
            is_abs = path_str.startswith("/")

    if not is_abs and make_absolute:
        path_str = os.path.join(os.getcwd(), path_str)
        is_abs = True

    parts: list[str] = []
    trailing_slash = False
    for part in path_str.split("/"):
        if part == "" or part == ".":
            trailing_slash = True
            continue
        if part == "..":
            trailing_slash = True
        else:
            trailing_slash = False
        parts.append(part)

    if not parts:
        result = "/" if is_abs else "."
    else:
        result = "/".join(parts)
        if is_abs:
            result = "/" + result
        if trailing_slash and preserve_dir:
            result = result + "/"

    if result == path_orig_str:
        # The result is still the same as the orignal string. Return
        # the original instance.
        return path_orig

    final_result: Any
    if isinstance(path_orig, pathlib.Path):
        final_result = (type(path_orig))(result)
    else:
        final_result = result

    return typing.cast(TPathNormPath, final_result)


def path_basedir(filename: str) -> tuple[str, str]:
    cwd = path_norm(".", cwd=os.getcwd())
    basedir = path_norm(os.path.dirname(filename), cwd=cwd)
    return cwd, basedir


@contextlib.contextmanager
def use_or_open(
    file: Union[str, pathlib.Path, typing.IO[str]],
    *,
    mode: Literal["r", "w"] = "r",
) -> typing.Generator[typing.IO[str], None, None]:
    f: typing.IO[str]
    if isinstance(file, (str, pathlib.Path)):
        f = open(file, mode=mode)
    else:
        f = file
    try:
        yield f
    finally:
        if f is not file:
            f.close()


def enum_convert(
    enum_type: type[E],
    value: Any,
    default: Optional[E] = None,
) -> E:

    if value is None:
        # We only allow None, if the caller also specified a default value.
        if default is not None:
            return default
    elif isinstance(value, enum_type):
        return value
    elif isinstance(value, int):
        try:
            return enum_type(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value} to {enum_type}")
    elif isinstance(value, str):
        v = value.strip()

        # Try lookup by name.
        try:
            return enum_type[v]
        except KeyError:
            pass

        # Try the string as integer value.
        try:
            return enum_type(int(v))
        except Exception:
            pass

        # Finally, try again with all upper case. Also, all "-" are replaced
        # with "_", but only if the result is unique.
        v2 = v.upper().replace("-", "_")
        matches = [e for e in enum_type if e.name.upper() == v2]
        if len(matches) == 1:
            return matches[0]

        raise ValueError(f"Cannot convert {value} to {enum_type}")

    raise ValueError(f"Invalid type for conversion to {enum_type}")


def enum_convert_list(
    enum_type: type[E],
    value: Any,
    *,
    default_range: Union[None, Iterable[E], _MISSING_TYPE] = MISSING,
) -> list[E]:
    output: list[E] = []

    if isinstance(value, str):
        for part in value.split(","):
            part = part.strip()
            if not part:
                # Empty words are silently skipped.
                continue

            cases: Optional[list[E]] = None

            # Try to parse as a single enum value.
            try:
                cases = [enum_convert(enum_type, part)]
            except Exception:
                cases = None

            if part == "*":
                if default_range is None:
                    # no special treatment of "*"
                    pass
                elif isinstance(default_range, _MISSING_TYPE):
                    # Shorthand for the entire range (sorted by numeric values)
                    cases = sorted(enum_type, key=lambda e: e.value)
                else:
                    if not isinstance(default_range, list):
                        # Ensure we iterate the input argument only once.
                        default_range = list(default_range)
                    cases = default_range

            if cases is None:
                # Could not be parsed as single entry. Try to parse as range.

                def _range_endpoint(s: str) -> int:
                    try:
                        return int(s)
                    except Exception:
                        pass
                    enum_val = enum_convert(enum_type, s)
                    v2 = int(enum_val.value)
                    if v2 != enum_val.value:
                        raise ValueError()
                    return v2

                try:
                    # Try to detect this as range. Both end points may either by
                    # an integer or an enum name.
                    #
                    # Note that since we use "-" to denote the range, we cannot have
                    # a range that involves negative enum values (otherwise, enum_convert()
                    # is fine to parse a single enum from a negative number in a string).
                    start, end = [_range_endpoint(s) for s in part.split("-")]
                except Exception:
                    # Couldn't parse as range.
                    pass
                else:
                    # We have a range.
                    cases = None
                    for i in range(start, end + 1):
                        try:
                            e = enum_convert(enum_type, i)
                        except Exception:
                            # When specifying a range, then missing enum values are
                            # silently ignored. Note that as a whole, the range may
                            # still not be empty.
                            continue
                        if cases is None:
                            cases = []
                        cases.append(e)

            if cases is None:
                raise ValueError(f"Invalid test case id: {part}")

            output.extend(cases)
    elif isinstance(value, list):
        for idx, part in enumerate(value):
            # First, try to parse the list entry with plain enum_convert.
            cases = None
            try:
                cases = [enum_convert(enum_type, part)]
            except Exception:
                # Now, try to parse as a list (but only if we have a string, no lists in lists).
                if isinstance(part, str):
                    try:
                        cases = enum_convert_list(enum_type, part)
                    except Exception:
                        pass
            if not cases:
                raise ValueError(
                    f'list at index {idx} contains invalid value "{part}" for enum {enum_type}'
                )
            output.extend(cases)
    else:
        raise ValueError(f"Invalid {enum_type} value of type {type(value)}")

    return output


def json_parse_list(jstr: str, *, strict_parsing: bool = False) -> list[Any]:
    try:
        lst = json.loads(jstr)
    except ValueError:
        if strict_parsing:
            raise
        return []

    if not isinstance(lst, list):
        if strict_parsing:
            raise ValueError("JSON data does not contain a list")
        return []

    return lst


def json_dump(
    data: Any,
    file: Union[str, pathlib.Path, typing.IO[str]],
) -> None:
    with use_or_open(file, mode="w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def dict_add_optional(vdict: dict[T1, T2], key: T1, val: Optional[T2]) -> None:
    if val is not None:
        vdict[key] = val


@typing.overload
def dict_get_typed(
    d: Mapping[Any, Any],
    key: Any,
    vtype: type[T],
    *,
    allow_missing: Literal[False] = False,
) -> T: ...


@typing.overload
def dict_get_typed(
    d: Mapping[Any, Any],
    key: Any,
    vtype: type[T],
    *,
    allow_missing: bool = False,
) -> Optional[T]: ...


def dict_get_typed(
    d: Mapping[Any, Any],
    key: Any,
    vtype: type[T],
    *,
    allow_missing: bool = False,
) -> Optional[T]:
    try:
        v = d[key]
    except KeyError:
        if allow_missing:
            return None
        raise KeyError(f'missing key "{key}"')
    if not isinstance(v, vtype):
        raise TypeError(f'key "{key}" expected type {vtype} but has value "{v}"')
    return v


def serialize_enum(
    data: Union[Enum, dict[Any, Any], list[Any], Any],
) -> Union[str, dict[Any, Any], list[Any], Any]:
    if isinstance(data, Enum):
        return data.name
    elif isinstance(data, dict):
        return {k: serialize_enum(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_enum(item) for item in data]
    else:
        return data


def dataclass_to_dict(obj: "DataclassInstance") -> dict[str, Any]:
    d = dataclasses.asdict(obj)
    return typing.cast(dict[str, Any], serialize_enum(d))


def dataclass_to_json(obj: "DataclassInstance") -> str:
    d = dataclass_to_dict(obj)
    return json.dumps(d)


# Takes a dataclass and the dict you want to convert from
# If your dataclass has a dataclass member, it handles that recursively
def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise ValueError(
            f"dataclass_from_dict() should only be used with dataclasses but is called with {cls}"
        )
    if not isinstance(data, dict):
        raise ValueError(
            f"requires a dictionary to in initialize dataclass {cls} but got {type(data)}"
        )
    for k in data:
        if not isinstance(k, str):
            raise ValueError(
                f"requires a strdict to in initialize dataclass {cls} but has key {type(k)}"
            )
    data = dict(data)
    create_kwargs = {}
    for field in fields(cls):
        if field.name not in data:
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                raise ValueError(
                    f'Missing mandatory argument "{field.name}" for dataclass {cls}'
                )
            continue

        if not field.init:
            continue

        def convert_simple(ck_type: Any, value: Any) -> Any:
            if is_dataclass(ck_type) and isinstance(value, dict):
                assert isinstance(ck_type, type)
                return dataclass_from_dict(ck_type, value)
            if (
                actual_type is None
                and isinstance(ck_type, type)
                and issubclass(ck_type, Enum)
            ):
                return enum_convert(ck_type, value)
            if ck_type is float and isinstance(value, int):
                return float(value)
            return value

        actual_type = typing.get_origin(field.type)

        value = data.pop(field.name)

        converted = False

        if actual_type is typing.Union:
            # This is an Optional[]. We already have a value, we check for the requested
            # type. check_type() already implements this, but we need to also check
            # it here, for the dataclass/enum handling below.
            args = typing.get_args(field.type)
            ck_type = None
            if len(args) == 2:
                NoneType = type(None)
                if args[0] is NoneType:
                    ck_type = args[1]
                elif args[1] is NoneType:
                    ck_type = args[0]
            if ck_type is not None:
                value_converted = convert_simple(ck_type, value)
                converted = True
        elif actual_type is list:
            args = typing.get_args(field.type)
            if isinstance(value, (list, tuple)) and len(args) == 1:
                value_converted = [convert_simple(args[0], v) for v in value]
                converted = True
        elif actual_type is tuple:
            args = typing.get_args(field.type)
            if isinstance(value, (list, tuple)) and len(args) == 2 and args[1] is ...:
                value_converted = tuple(convert_simple(args[0], v) for v in value)
                converted = True

        if not converted:
            value_converted = convert_simple(field.type, value)

        if not check_type(value_converted, field.type):
            raise TypeError(
                f"Expected type '{field.type}' for attribute '{field.name}' but received type '{type(value)}' ({value})"
            )

        create_kwargs[field.name] = value_converted

    if data:
        raise ValueError(
            f"There are left over keys {list(data)} to create dataclass {cls}"
        )

    result: Any = cls(**create_kwargs)
    return typing.cast(T, result)


def dataclass_from_json(cls: type[T], jsondata: str) -> T:
    data = json.loads(jsondata)
    return dataclass_from_dict(cls, data)


def dataclass_from_file(
    cls: type[T],
    file: Union[str, pathlib.Path, typing.IO[str]],
) -> T:
    with use_or_open(file) as f:
        data = json.load(f)
    return dataclass_from_dict(cls, data)


def check_type(
    value: typing.Any,
    type_hint: typing.Any,
) -> bool:

    # Some naive type checking. This is used for ensuring that data classes
    # contain the expected types (see @strict_dataclass).
    #
    # That is most interesting, when we initialize the data class with
    # data from an untrusted source (like elements from a JSON parser).

    actual_type = typing.get_origin(type_hint)
    if actual_type is None:
        if isinstance(type_hint, str):
            raise NotImplementedError(
                f'Type hint "{type_hint}" as string is not implemented by check_type()'
            )

        if type_hint is typing.Any:
            return True
        return isinstance(value, typing.cast(Any, type_hint))

    if actual_type is typing.Union:
        args = typing.get_args(type_hint)
        return any(check_type(value, a) for a in args)

    if actual_type is list:
        args = typing.get_args(type_hint)
        (arg,) = args
        return isinstance(value, list) and all(check_type(v, arg) for v in value)

    if actual_type is dict or actual_type is Mapping:
        args = typing.get_args(type_hint)
        (arg_key, arg_val) = args
        return isinstance(value, dict) and all(
            check_type(k, arg_key) and check_type(v, arg_val) for k, v in value.items()
        )

    if actual_type is tuple:
        # https://docs.python.org/3/library/typing.html#annotating-tuples
        if not isinstance(value, tuple):
            return False
        args = typing.get_args(type_hint)
        if len(args) == 1 and args[0] == ():
            # This is an empty tuple tuple[()].
            return len(value) == 0
        if len(args) == 2 and args[1] is ...:
            # This is a tuple[T, ...].
            return all(check_type(v, args[0]) for v in value)
        return len(value) == len(args) and all(
            check_type(v, args[idx]) for idx, v in enumerate(value)
        )

    raise NotImplementedError(
        f'Type hint "{type_hint}" with origin type "{actual_type}" is not implemented by check_type()'
    )


def dataclass_check(
    instance: "DataclassInstance",
    *,
    with_post_check: bool = True,
) -> None:

    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if not check_type(value, field.type):
            raise TypeError(
                f"Expected type '{field.type}' for attribute '{field.name}' but received type '{type(value)}' ({value})"
            )

    if with_post_check:
        # Normally, data classes support __post_init__(), which is called by __init__()
        # already. Add a way for a @strict_dataclass to add additional validation *after*
        # the original check.
        _post_check = getattr(type(instance), "_post_check", None)
        if _post_check is not None:
            _post_check(instance)


def strict_dataclass(cls: TCallable) -> TCallable:

    init = getattr(cls, "__init__")

    def wrapped_init(self: Any, *args: Any, **argv: Any) -> None:
        init(self, *args, **argv)
        dataclass_check(self)

    setattr(cls, "__init__", wrapped_init)
    return cls


def yamlpath_build(
    yamlpath: str,
    yamlidx: Optional[int] = None,
    key: Optional[str] = None,
    subpath: Optional[str] = None,
) -> str:
    if yamlidx is not None:
        yamlpath += f"[{yamlidx}]"
    if key is not None:
        yamlpath += f".{key}"
    if subpath is not None:
        yamlpath += subpath
    return yamlpath


def yamlpath_value_error(
    yamlpath: str,
    msg: str,
    *,
    yamlidx: Optional[int] = None,
    key: Optional[str] = None,
    subpath: Optional[str] = None,
) -> ValueError:
    yamlpath = yamlpath_build(
        yamlpath,
        yamlidx=yamlidx,
        key=key,
        subpath=subpath,
    )
    return ValueError(f'"{yamlpath}": {msg}')


@dataclass(frozen=True)
class StructParseParseContext:
    arg: Any
    yamlpath: str = dataclasses.field(default="")
    yamlidx: int = dataclasses.field(default=0)

    @contextlib.contextmanager
    def with_strdict(self) -> typing.Generator["StructParseVarg", None, None]:
        return _structparse_with_strdict(self.arg, self.yamlpath)

    @staticmethod
    def enumerate_list(
        yamlpath: str,
        arg: Iterable[Any],
    ) -> list["StructParseParseContext"]:
        return [
            StructParseParseContext(
                arg2,
                yamlpath=f"{yamlpath}[{yamlidx2}]",
                yamlidx=yamlidx2,
            )
            for yamlidx2, arg2 in enumerate(arg)
        ]

    def build_yamlpath(
        self,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> str:
        return yamlpath_build(
            self.yamlpath,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )

    def value_error(
        self,
        msg: str,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> ValueError:
        return yamlpath_value_error(
            self.yamlpath,
            msg,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )


@dataclass(frozen=True)
class StructParsePopContext:
    vdict: dict[str, Any]
    base_yamlpath: str
    key: str

    @staticmethod
    def for_name(
        vdict: dict[str, Any],
        yamlpath: str,
        key: str = "name",
    ) -> "StructParsePopContext":
        return StructParsePopContext(vdict, yamlpath, key)

    def pop_value(self) -> Optional[Any]:
        return self.vdict.pop(self.key, None)

    @property
    def yamlpath(self) -> str:
        return yamlpath_build(self.base_yamlpath, key=self.key)

    def build_yamlpath(
        self,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> str:
        return yamlpath_build(
            self.yamlpath,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )

    def value_error(
        self,
        msg: str,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> ValueError:
        return yamlpath_value_error(
            self.yamlpath,
            msg,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )


@dataclass(frozen=True)
class StructParseVarg:
    vdict: dict[str, Any]
    yamlpath: str
    check_empty: bool = dataclasses.field(default=True, init=False)

    def for_key(self, key: str) -> StructParsePopContext:
        """
        Returns a StructParsePopContext that contains [vdict, yamlpath, key].

        The structparse_pop_*() functions have a StructParsePopContext argument
        with the parameters of what they parse.

        Example:

           foo = structparse_pop_str(varg.for_key("foo"))
        """
        return StructParsePopContext(self.vdict, self.yamlpath, key)

    def for_name(self, key: str = "name") -> StructParsePopContext:
        """
        Same as for_key(), but defaults to a key "name".

        Example:

           foo = structparse_pop_str_name(varg.for_name())
        """
        return self.for_key(key)

    def skip_check_empty(self) -> None:
        """
        With structparse_with_strdict(), indicate that the final
        structparse_check_empty_dict() should be skipped.
        """
        object.__setattr__(self, "check_empty", False)


def structparse_check_strdict(arg: Any, yamlpath: str) -> dict[str, Any]:
    """
    Checks that "args" is a strdict and returns a shallow copy
    of the dictionary.

    The usage is then to pop keys from the dictionary (structparse_pop_*())
    and at the end check that no more (unknown) keys are left (structparse_check_empty_dict()).
    """
    if not isinstance(arg, dict):
        raise ValueError(f'"{yamlpath}": expects a dictionary but got {type(arg)}')
    for k, v in arg.items():
        if not isinstance(k, str):
            raise ValueError(
                f'"{yamlpath}": expects all dictionary keys to be strings but got {type(k)}'
            )

    # We shallow-copy the dictionary, because the caller will remove entries
    # to find unknown entries (see _check_empty_dict()).
    return dict(arg)


def structparse_check_empty_dict(vdict: dict[str, Any], yamlpath: str) -> None:
    """
    Checks that "vdict" is empty or fail with an exception.

    The usage is to first shallow copy the dictionary (with structparse_check_strdict()),
    the pop all known keys (structparse_pop_*()), and finally check that no (unknown)
    keys are left. Possibly use this via "with structparse_with_strdict() as varg".
    """
    length = len(vdict)
    if length == 1:
        raise ValueError(f'"{yamlpath}": unknown key {repr(list(vdict)[0])}')
    if length > 1:
        raise ValueError(f'"{yamlpath}": unknown keys {list(vdict)}')


@contextlib.contextmanager
def structparse_with_strdict(
    arg: Any,
    yamlpath: str,
) -> typing.Generator[StructParseVarg, None, None]:
    """
    Context manager for parsing a strdict.

    arg: the argument, which is validated to be a string dictinary.
    yamlpath: the YAML path for "arg".

    Example:

        with structparse_with_strdict(arg, yamlpath) as varg:
            name = structparse_pop_str_name(varg.for_name())
            foo = structparse_pop_int(varg.for_key("foo"), default=None)

    Above is basically the same as

        vdict = structparse_check_strdict(arg, yamlpath)
        name = structparse_pop_str_name(vdict, yamlpath, "name")
        foo = structparse_pop_int(vdict, yamlpath, "foo", default=None)
        structparse_check_empty_dict(vdict)
    """
    return _structparse_with_strdict(arg, yamlpath)


def _structparse_with_strdict(
    arg: Any,
    yamlpath: str,
) -> typing.Generator[StructParseVarg, None, None]:
    vdict = structparse_check_strdict(arg, yamlpath)
    varg = StructParseVarg(vdict, yamlpath)
    yield varg
    if varg.check_empty:
        structparse_check_empty_dict(vdict, yamlpath)


def structparse_pop_str(
    pargs: StructParsePopContext,
    *,
    default: Union[TOptionalStr, _MISSING_TYPE] = MISSING,
    empty_as_default: Optional[bool] = None,
    allow_empty: Optional[bool] = None,
    check: Optional[typing.Callable[[str], bool]] = None,
    check_missing: Optional[typing.Callable[[StructParsePopContext], None]] = None,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, str], None]] = None,
    check_regex: Optional[Union[str, re.Pattern[str]]] = None,
) -> Union[str, TOptionalStr]:
    """
    Pop "key" from "vdict", validates that it's a string and returns it.
    If "default" is given it is returned for missing keys. Otherwise,
    the key is mandatory.
    """
    # The arguments allow to carefully control what happens with empty
    # values. Usually, most parameters are unset by the caller, so we
    # must determine their actual values depending on the parameters
    # we have. It's chosen in a way, so that it makes the most sense
    # for the caller.
    if allow_empty is None:
        # "allow_empty" will default to True, if "empty_as_default" is set.
        if empty_as_default is not None:
            allow_empty = True
    if empty_as_default is None:
        # "empty_as_default" defaults to True, if "allow_empty" is True.
        empty_as_default = allow_empty is None or not allow_empty
    if allow_empty is None:
        # At this point, if "allow_empty" is still undecided, we allow
        # it if "empty_as_default" or if we have a "check".
        allow_empty = empty_as_default or check is not None
    if not allow_empty:
        empty_as_default = False

    v = pargs.pop_value()
    if v is not None and not isinstance(v, str):
        raise pargs.value_error(f"expects a string but got {repr(v)}")
    if v is None or (not v and empty_as_default):
        if check_missing is not None:
            check_missing(pargs)
        if isinstance(default, _MISSING_TYPE):
            raise pargs.value_error("mandatory key missing")
        return default
    if not v:
        if not allow_empty:
            raise pargs.value_error("cannot be an empty string")

    if check is not None:
        if not check(v):
            raise pargs.value_error("invalid string")

    if check_ctx is not None:
        check_ctx(pargs, v)

    if check_regex is not None:
        pattern_re = as_regex(check_regex)
        if not pattern_re.search(v):
            raise pargs.value_error(
                f"does not match pattern {repr(pattern_re.pattern)}"
            )

    return typing.cast(TOptionalStr, v)


def structparse_pop_str_name(
    pargs: StructParsePopContext,
    *,
    default: Union[TOptionalStr, _MISSING_TYPE] = MISSING,
    check: Optional[typing.Callable[[str], bool]] = None,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, str], None]] = None,
    check_regex: Optional[Union[str, re.Pattern[str]]] = None,
) -> Union[str, TOptionalStr]:
    return structparse_pop_str(
        pargs,
        default=default,
        allow_empty=False,
        check=check,
        check_ctx=check_ctx,
        check_regex=check_regex,
    )


def structparse_pop_int(
    pargs: StructParsePopContext,
    *,
    default: Union[TOptionalInt, _MISSING_TYPE] = MISSING,
    check: Optional[typing.Callable[[int], bool]] = None,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, float], None]] = None,
    description: str = "a number",
) -> Union[int, TOptionalInt]:
    v = pargs.pop_value()
    if v is None:
        if isinstance(default, _MISSING_TYPE):
            raise pargs.value_error(f"requires {description}")
        return default
    try:
        val = int(v)
    except Exception:
        raise pargs.value_error(f"expects {description} but got {repr(v)}")

    if check is not None:
        if not check(val):
            raise pargs.value_error(f"expects {description} but got {repr(v)}")

    if check_ctx is not None:
        check_ctx(pargs, val)

    return val


def structparse_pop_float(
    pargs: StructParsePopContext,
    *,
    default: Union[TOptionalFloat, _MISSING_TYPE] = MISSING,
    check: Optional[typing.Callable[[float], bool]] = None,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, float], None]] = None,
    description: str = "a floating point number",
) -> Union[float, TOptionalFloat]:
    v = pargs.pop_value()
    if v is None:
        if isinstance(default, _MISSING_TYPE):
            raise pargs.value_error(
                f"requires {description}",
            )
        return default
    try:
        val = float(v)
    except Exception:
        raise pargs.value_error(f"expects {description} but got {repr(v)}")

    if check is not None:
        if not check(val):
            raise pargs.value_error(f"expects {description} but got {repr(v)}")

    if check_ctx is not None:
        check_ctx(pargs, val)

    return val


def structparse_pop_bool(
    pargs: StructParsePopContext,
    *,
    default: Union[TOptionalBool, _MISSING_TYPE] = MISSING,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, float], None]] = None,
    description: str = "a boolean",
) -> Union[bool, TOptionalBool]:
    has_default = not isinstance(default, _MISSING_TYPE)
    v = pargs.pop_value()
    try:
        # Just like str_to_bool(), we accept "", "default", and "-1" as default
        # values (if `default` is not MISSING).
        val = str_to_bool(v, on_default=None if has_default else MISSING)
    except Exception:
        if v is None:
            raise pargs.value_error(f"requires {description}")
        raise pargs.value_error(f"expects {description} but got {repr(v)}")

    if val is None:
        assert not isinstance(default, _MISSING_TYPE)
        return default

    # Like other check_ctx() callbacks, we don't call them for a
    # default value.
    if check_ctx is not None:
        check_ctx(pargs, val)

    return val


@typing.overload
def structparse_pop_enum(
    pargs: StructParsePopContext,
    *,
    enum_type: type[E],
    default: Union[E, _MISSING_TYPE] = MISSING,
) -> E: ...


@typing.overload
def structparse_pop_enum(
    pargs: StructParsePopContext,
    *,
    enum_type: type[E],
    default: Literal[None],
) -> Optional[E]: ...


def structparse_pop_enum(
    pargs: StructParsePopContext,
    *,
    enum_type: type[E],
    default: Union[Optional[E], _MISSING_TYPE] = MISSING,
) -> Optional[E]:
    v = pargs.pop_value()
    if v is None:
        if isinstance(default, _MISSING_TYPE):
            raise pargs.value_error(
                f"requires one of {', '.join(e.name for e in enum_type)}"
            )
        return default
    if isinstance(default, _MISSING_TYPE):
        default = None
    try:
        return enum_convert(enum_type, v, default=default)
    except Exception:
        raise pargs.value_error(
            f"requires one of {', '.join(e.name for e in enum_type)} but got {repr(v)}"
        )


def structparse_pop_list(
    pargs: StructParsePopContext,
    *,
    allow_missing: Optional[bool] = None,
    allow_empty: bool = True,
) -> list[Any]:
    """
    Checks that "key" is a list (of anything) and returns a shallow copy of the
    list. This always returns a (potentially empty) list. By default, missing
    key and empty list is allowed, but that can be restricted with the
    "allow_missing" and "allow_empty" parameters.
    """
    if allow_missing is None:
        allow_missing = allow_empty
    v = pargs.pop_value()
    if v is None:
        if not allow_missing:
            raise pargs.value_error("mandatory list argument missing")
        # We never return None here. For many callers that is what we just
        # want. For callers that want to do something specific if the key is
        # unset, they should check first whether vdict contains the key.
        return []
    if not isinstance(v, list):
        raise pargs.value_error(f"requires a list but got {type(v)}")
    if not v:
        if not allow_empty:
            raise pargs.value_error("list cannot be empty")
    # Return a shallow copy of the list.
    return list(v)


def structparse_pop_obj(
    pargs: StructParsePopContext,
    *,
    construct: typing.Callable[[StructParseParseContext], T],
    default: Union[T2, _MISSING_TYPE] = MISSING,
    check_ctx: Optional[typing.Callable[[StructParsePopContext, T], None]] = None,
    construct_default: bool = False,
) -> Union[T, T2]:
    """
    Pops "key" from "vdict" and passes it to "construct" callback to parse
    and construct a result.

    By default, the key is mandatory. If "construct_default" is True,
    for missing keys we pass None to "construct". This allows to generate
    the callback a default value. Otherwise, if "default" is set, that
    is returned for missing keys.
    """
    v = pargs.pop_value()
    is_default = v is None
    if not construct_default and is_default:
        if isinstance(default, _MISSING_TYPE):
            raise pargs.value_error("mandatory key missing")
        return default

    pctx = StructParseParseContext(v, pargs.yamlpath)

    obj = construct(pctx)

    if not is_default and check_ctx is not None:
        check_ctx(pargs, obj)

    return obj


def structparse_pop_objlist(
    pargs: StructParsePopContext,
    *,
    construct: typing.Callable[[StructParseParseContext], T],
    allow_missing: Optional[bool] = None,
    allow_empty: bool = True,
) -> tuple[T, ...]:
    v = structparse_pop_list(
        pargs,
        allow_missing=allow_missing,
        allow_empty=allow_empty,
    )

    pctxes = StructParseParseContext.enumerate_list(pargs.yamlpath, v)
    return tuple(construct(pctx) for pctx in pctxes)


@typing.overload
def structparse_pop_objlist_to_dict(
    pargs: StructParsePopContext,
    *,
    construct: typing.Callable[[StructParseParseContext], TStructParseBaseNamed],
    get_key: Literal[None] = None,
    allow_empty: bool = True,
    allow_duplicates: bool = False,
) -> dict[str, TStructParseBaseNamed]: ...


@typing.overload
def structparse_pop_objlist_to_dict(
    pargs: StructParsePopContext,
    *,
    construct: typing.Callable[[StructParseParseContext], T],
    get_key: typing.Callable[[T], T2],
    allow_empty: bool = True,
    allow_duplicates: bool = False,
) -> dict[T2, T]: ...


def structparse_pop_objlist_to_dict(
    pargs: StructParsePopContext,
    *,
    construct: typing.Callable[[StructParseParseContext], T],
    get_key: Optional[typing.Callable[[T], T2]] = None,
    allow_empty: bool = True,
    allow_duplicates: bool = False,
) -> dict[T2, T]:
    lst = structparse_pop_objlist(
        pargs,
        construct=construct,
        allow_empty=allow_empty,
    )
    result: dict[T2, tuple[int, T]] = {}
    for yamlidx2, item in enumerate(lst):
        if get_key is not None:
            item_key = get_key(item)
        else:
            if not isinstance(item, StructParseBaseNamed):
                raise RuntimeError(
                    f"list requires StructParseBaseNamed elements but we got {type(item)}"
                )
            item_key = typing.cast("T2", item.name)
        item2 = result.get(item_key, None)
        if item2 is not None:
            if allow_duplicates:
                # We allow duplicates. Last occurrence wins. We remove the old
                # entry (because the dict is sorted, and we want to preserve the
                # order.
                del result[item_key]
            else:
                if isinstance(item_key, Enum):
                    key_name = item_key.name
                else:
                    key_name = repr(item_key)
                raise pargs.value_error(
                    f'duplicate key {repr(key_name)} with "{pargs.build_yamlpath(yamlidx=item2[0])}"',
                    yamlidx=yamlidx2,
                )
        result[item_key] = (yamlidx2, item)
    return {k: v[1] for k, v in result.items()}


class DeferredReference:
    _lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def init(self, val: Any) -> None:
        ref: Optional[weakref.ref[Any]]
        if val is None:
            ref = None
        else:
            ref = weakref.ref(val)
        with self._lock:
            if hasattr(self, "_ref"):
                raise RuntimeError("Reference can only be initialized once")
            setattr(self, "_ref", ref)

    def get(self, typ: type[T]) -> T:
        ref: Optional[weakref.ref[Any]]
        ref2: Any
        with self._lock:
            try:
                ref = getattr(self, "_ref")
            except AttributeError:
                raise RuntimeError("Reference not set") from None
        if ref is None:
            ref2 = None
        else:
            ref2 = ref()
            if ref2 is None:
                raise RuntimeError("Reference already destroyed")
        if not isinstance(ref2, typ):
            raise RuntimeError("Reference has unexpected type")
        return ref2


@strict_dataclass
@dataclass(frozen=True, **KW_ONLY_DATACLASS)
class StructParseBase(abc.ABC):
    yamlpath: str
    yamlidx: int

    _owner_reference: DeferredReference = dataclasses.field(
        default_factory=DeferredReference,
        init=False,
        repr=False,
        compare=False,
    )

    @abc.abstractmethod
    def serialize(self) -> Union[dict[str, Any], list[Any]]:
        pass

    def serialize_json(self) -> str:
        return json.dumps(self.serialize())

    def build_yamlpath(
        self,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> str:
        return yamlpath_build(
            self.yamlpath,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )

    def value_error(
        self,
        msg: str,
        *,
        yamlidx: Optional[int] = None,
        key: Optional[str] = None,
        subpath: Optional[str] = None,
    ) -> ValueError:
        return yamlpath_value_error(
            self.yamlpath,
            msg,
            yamlidx=yamlidx,
            key=key,
            subpath=subpath,
        )


@strict_dataclass
@dataclass(frozen=True, **KW_ONLY_DATACLASS)
class StructParseBaseNamed(StructParseBase, abc.ABC):
    name: str

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
        }


def repeat_for_same_result(fcn: TCallable) -> TCallable:
    # This decorator wraps @fcn and will call it (up to 20 times) until the
    # same result was returned twice in a row. The purpose is when we fetch
    # several pieces of information form the system, that can change at any
    # time. We would like to get a stable, self-consistent result.
    @functools.wraps(fcn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        has_any_result = False
        result = None
        has_previous_result = False
        last_exception = None
        for i in range(20):
            try:
                new_result = fcn(*args, **kwargs)
            except Exception as e:
                has_previous_result = False
                last_exception = e
                continue
            if has_previous_result and result == new_result:
                return new_result
            result = new_result
            has_any_result = True
            has_previous_result = True

        if not has_any_result:
            raise unwrap(last_exception)

        # We didn't get a stable result after 20 tries. Return
        # the result that we got.
        return result

    return typing.cast(TCallable, wrapped)


def etc_hosts_update_data(
    content: str,
    new_entries: Mapping[str, tuple[str, Optional[Iterable[str]]]],
) -> str:

    lineregex = re.compile(r"^\s*[a-fA-F0-9:.]+\s+([-a-zA-Z0-9_.]+)(\s+.*)?$")

    def _unpack(
        v: tuple[str, Optional[Iterable[str]]],
    ) -> Union[Literal[False], tuple[str, tuple[str, ...]]]:
        n, a = v
        if a is None:
            a = ()
        else:
            a = tuple(a)
        return n, a

    entries = {k: _unpack(v) for k, v in new_entries.items()}

    def _build_line(name: str, ipaddr: str, aliases: tuple[str, ...]) -> str:
        if aliases:
            s_aliases = f" {' '.join(aliases)}"
        else:
            s_aliases = ""
        return f"{ipaddr} {name}{s_aliases}"

    result = []
    for line in content.splitlines():
        m = lineregex.search(line)
        if m:
            name = m.group(1)
            entry = entries.get(name)
            if entry is None:
                pass
            elif entry is False:
                continue
            else:
                line = _build_line(name, *entry)
                entries[name] = False
        result.append(line)

    entries2 = [(k, v) for k, v in entries.items() if v is not False]
    if entries2:
        if result and result[-1] != "":
            result.append("")
        for name, entry in entries2:
            result.append(_build_line(name, *entry))

    if not result:
        return ""

    result.append("")
    return "\n".join(result)


def etc_hosts_update_file(
    new_entries: Mapping[str, tuple[str, Optional[Iterable[str]]]],
    filename: PathType = "/etc/hosts",
) -> str:
    try:
        with open(filename, "rb") as f:
            b_content = f.read()
    except Exception:
        b_content = b""

    new_content = etc_hosts_update_data(
        b_content.decode("utf-8", errors="surrogateescape"),
        new_entries,
    )

    with open(filename, "wb") as f:
        f.write(new_content.encode("utf-8", errors="surrogateescape"))

    return new_content


class Serial:
    def __init__(self, port: str, baudrate: int = 115200):
        try:
            import serial
        except ImportError as e:
            raise ImportError(
                "pyserial is required for the Serial class. "
                "Install with: pip install 'ktoolbox[pyserial]'"
            ) from e

        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0)
        self._bin_buf = b""
        self._str_buf: Optional[str] = None

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

    def send(self, msg: str, *, sleep: float = 1) -> None:
        logger.debug(f"serial[{self.port}]: send {repr(msg)}")
        self._ser.write(msg.encode("utf-8", errors="surrogateescape"))
        if sleep > 0:
            self.expect(pattern=None, timeout=sleep)

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
                if not self._bin_buf:
                    self._str_buf = s
                elif (
                    self._str_buf
                    and not char_is_surrogateescaped(self._str_buf[-1])
                    and not char_is_surrogateescaped(s[0])
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

    @typing.overload
    def expect(
        self,
        pattern: Union[str, re.Pattern[str]],
        timeout: float = 30,
    ) -> str: ...

    @typing.overload
    def expect(
        self,
        pattern: None,
        timeout: float = 30,
    ) -> None: ...

    @typing.overload
    def expect(
        self,
        pattern: Optional[Union[str, re.Pattern[str]]],
        timeout: float = 30,
    ) -> Optional[str]: ...

    def expect(
        self,
        pattern: Optional[Union[str, re.Pattern[str]]],
        timeout: float = 30,
    ) -> Optional[str]:
        import select

        end_timestamp = time.monotonic() + timeout

        # We use DOTALL like pexpect does.
        # If you need something else, compile the pattern yourself.
        #
        # See also https://pexpect.readthedocs.io/en/stable/overview.html#find-the-end-of-line-cr-lf-conventions
        pattern_re = as_regex(pattern, flags=re.DOTALL)

        if pattern_re is not None:
            logger.debug(f"serial[{self.port}]: expect message {repr(pattern)}")

        while True:
            self.read_all()

            if pattern_re is not None:
                buffer = self.buffer
                matches = re.finditer(pattern_re, buffer)
                for match in matches:
                    end_idx = match.end()
                    consumed_chars = buffer[:end_idx]
                    consumed_bytes = consumed_chars.encode(
                        "utf-8",
                        errors="surrogateescape",
                    )
                    assert self._bin_buf.startswith(consumed_bytes)
                    logger.debug(
                        f"serial[{self.port}]: found expected message {len(consumed_bytes)} bytes, {len(self._bin_buf) - len(consumed_bytes)} bytes remaning"
                    )
                    self._str_buf = buffer[end_idx:]
                    self._bin_buf = self._bin_buf[len(consumed_bytes) :]
                    return consumed_chars

            remaining_time = end_timestamp - time.monotonic()
            if remaining_time <= 0:
                if pattern_re is not None:
                    s = self._bin_buf.decode("utf-8", errors="surrogateescape")
                    logger.debug(
                        f"serial[{self.port}]: did not find expected message {repr(pattern)} (buffer content is {repr(s)})"
                    )
                    raise RuntimeError(
                        f"Did not receive expected message {repr(pattern)} within timeout (buffer content is {repr(s)})"
                    )
                return None
            _, _, _ = select.select([self._ser], [], [], remaining_time)

    def __enter__(self) -> "Serial":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional["TracebackType"],
    ) -> None:
        self._ser.close()


def _log_parse_level_str(lvl: str) -> Optional[int]:
    lvl2 = lvl.lower().strip()
    if lvl2:
        DISABLED = logging.FATAL + 10
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARN,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
            "fatal": logging.FATAL,
            "off": DISABLED,
            "none": DISABLED,
            "disabled": DISABLED,
        }
        if lvl2 in log_levels:
            return log_levels[lvl2]
    return None


@functools.cache
def get_program_epoch() -> int:
    return int(time.time())


@functools.cache
def get_program_appname(*, with_pysuffix: bool = True) -> str:
    name: Optional[str] = None

    if sys.argv:
        argv0 = sys.argv[0]
        if argv0 and argv0 not in ("", "-", "-c"):
            name = os.path.basename(argv0)

    if not name:
        import __main__

        main_file = getattr(__main__, "__file__", None)
        if main_file:
            name = os.path.basename(main_file)

    if not name:
        if sys.stdin.isatty():
            return "interactive"
        return "unknown"

    if not with_pysuffix:
        if name.endswith(".py"):
            name = name[:-3]

    return name


def log_parse_level(
    lvl: Optional[Union[int, bool, str]],
    *,
    default_level: Union[int, TOptionalInt] = logging.INFO,
    prefer_global_default: bool = True,
) -> Union[int, TOptionalInt]:
    if lvl is None or (isinstance(lvl, str) and lvl.lower().strip() == ""):
        if prefer_global_default:
            v = log_default_level()
            if v is not None:
                return v
        return default_level
    if isinstance(lvl, bool):
        return logging.DEBUG if lvl else logging.INFO
    if isinstance(lvl, int):
        return lvl
    if isinstance(lvl, str):
        v = _log_parse_level_str(lvl)
        if v is not None:
            return v
    raise ValueError(f"invalid log level {repr(lvl)}")


@functools.cache
def log_all_loggers() -> bool:
    # By default, the main application calls common.log_config_logger()
    # and configures only certain loggers ("myapp", "ktoolbox"). If
    # KTOOLBOX_ALL_LOGGERS is set to True, then instead the root logger
    # will be configured which may affect also other python modules.
    return str_to_bool(getenv_config("KTOOLBOX_ALL_LOGGERS"), False)


@functools.cache
def log_default_level() -> Optional[int]:
    # On the command line, various main programs allow to specify the log
    # level. If they leave it unspecified, the default can be configured via
    # "KTOOLBOX_LOGLEVEL" environment variable. If still unspecified, the
    # default is determined by the main application that calls
    # common.log_config_logger().
    v = getenv_config("KTOOLBOX_LOGLEVEL")
    if v is not None:
        return _log_parse_level_str(v)
    return None


def _env_get_ktoolbox_logfile_parse(
    v: str,
) -> Optional[tuple[str, Optional[int], bool]]:
    if not v:
        return None
    append = True
    level: Optional[int] = None
    if ":" in v:
        s_level, v1 = v.split(":", 1)
        try:
            level = log_parse_level(
                s_level,
                prefer_global_default=False,
                default_level=None,
            )
        except ValueError:
            # If this is not a valid log-level, we assume that it is part of
            # the filename.
            pass
        else:
            if not v1:
                return None
            v = v1
    if v[0] in ("+", "="):
        append = v[0] == "+"
        v = v[1:]
        if not v:
            return None

    # Supports % substitutions similar to /proc/sys/kernel/core_pattern.
    substitutions: dict[str, typing.Callable[[], str]] = {
        "%p": lambda: str(os.getpid()),
        "%h": lambda: socket.gethostname().split(".", 1)[0],
        "%t": lambda: str(get_program_epoch()),
        "%a": lambda: get_program_appname(with_pysuffix=False),
        "%%": lambda: "%",
    }

    def _replace(match: re.Match[str]) -> str:
        match_str = match.group(0)
        if match_str not in substitutions:
            return match_str

        val = (substitutions[match_str])()

        # Freeze the value that we fetched (so that multiple replacements are
        # guaranteed to give the same result).
        substitutions[match_str] = lambda: val

        return val

    v = re.sub("%.", _replace, v)

    return (v, level, append)


@functools.cache
def _env_get_ktoolbox_logfile() -> Optional[tuple[str, Optional[int], bool]]:
    v = getenv_config("KTOOLBOX_LOGFILE")
    if not v:
        return None
    logfile = _env_get_ktoolbox_logfile_parse(v)
    if logfile is None:
        return None
    logfile_file, logfile_level, logfile_append = logfile
    logfile_file = path_norm(logfile_file, make_absolute=True)
    return (logfile_file, logfile_level, logfile_append)


@functools.cache
def _env_get_ktoolbox_logstdout() -> bool:
    v = getenv_config("KTOOLBOX_LOGSTDOUT")
    if v:
        v = v.strip().lower()
        v_bool = str_to_bool(v, on_error=None)
        if v_bool is not None:
            return v_bool
        if v in ("out", "stdout"):
            return True

    # Everything else is "stderr"
    return False


@functools.cache
def _env_get_ktoolbox_logtag() -> str:
    v = getenv_config("KTOOLBOX_LOGTAG")
    if not v:
        return ""
    return f"{v.replace('%', '%%')} "


def _log_create_formatter() -> logging.Formatter:
    logtag = _env_get_ktoolbox_logtag()
    fmt = (
        f"%(asctime)s.%(msecs)03d %(levelname)-7s {logtag}[th:%(thread)s]: %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"
    return logging.Formatter(fmt, datefmt)


if typing.TYPE_CHECKING:
    # https://github.com/python/cpython/issues/92128#issue-1222296106
    # https://github.com/python/typeshed/pull/5954#issuecomment-1114270968
    # https://mypy.readthedocs.io/en/stable/runtime_troubles.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
    _LogStreamHandler = logging.StreamHandler[typing.TextIO]
else:
    _LogStreamHandler = logging.StreamHandler


def _logHandlerSetLevelWithLock(handler: logging.Handler, *, level: int) -> None:
    handler.acquire()
    try:
        handler.setLevel(level)
    finally:
        handler.release()


class _LogHandlerStream(_LogStreamHandler):
    def __init__(self, level: int):
        stream = sys.stdout if _env_get_ktoolbox_logstdout() else sys.stderr
        super().__init__(stream)
        self.setLevel(level)
        self.setFormatter(_log_create_formatter())


class _LogHandlerFile(logging.FileHandler):
    def __init__(self, filename: str, *, append: bool, level: int):
        filename = path_norm(filename, make_absolute=True)
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        mode = "a" if append else "w"
        super().__init__(filename, mode)
        self.setLevel(level)
        self.setFormatter(_log_create_formatter())


def _logHandler_attach(
    logHandlerType: Union[type[_LogHandlerStream], type[_LogHandlerFile]],
    logger: logging.Logger,
    *,
    level: int,
) -> None:
    if logHandlerType is _LogHandlerFile:
        logfile = _env_get_ktoolbox_logfile()
        if logfile is None:
            return

    handler = iter_get_first(
        h for h in logger.handlers if isinstance(h, logHandlerType)
    )

    if handler is None:
        if logHandlerType is _LogHandlerFile:
            logfile_file, logfile_level, logfile_append = unwrap(logfile)
            if logfile_level is None:
                logfile_level = level
            handler = _LogHandlerFile(
                logfile_file,
                append=logfile_append,
                level=logfile_level,
            )
        else:
            assert logHandlerType is _LogHandlerStream
            handler = _LogHandlerStream(level=level)
        is_new_handler = True
    else:
        is_new_handler = False

    if is_new_handler:
        logger.addHandler(handler)
    else:
        _logHandlerSetLevelWithLock(handler, level=level)


def log_config_logger(
    level: Optional[Union[int, bool, str]],
    *loggers: Union[str, logging.Logger],
    default_level: int = logging.INFO,
) -> None:
    level = log_parse_level(level, default_level=default_level)

    if log_all_loggers():
        # If the environment variable KTOOLBOX_ALL_LOGGERS is True,
        # we configure the root logger instead.
        loggers = ("",)

    logger_level = level
    logfile = _env_get_ktoolbox_logfile()
    if logfile is not None:
        logfile_file, logfile_level, logfile_append = unwrap(logfile)
        if logfile_level is not None:
            logger_level = min(logger_level, logfile_level)

    for logger in loggers:
        real_logger = ExtendedLogger.unwrap(logger)
        with common_lock:
            _logHandler_attach(_LogHandlerStream, real_logger, level=level)
            _logHandler_attach(_LogHandlerFile, real_logger, level=level)
            real_logger.setLevel(logger_level)


def log_argparse_add_argument_verbose(parser: "argparse.ArgumentParser") -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
        help="Enable debug logging (overwrites KTOOLBOX_LOGLEVEL environment). Set KTOOLBOX_ALL_LOGGERS to configure all loggers.",
    )


def log_argparse_add_argument_verbosity(
    parser: "argparse.ArgumentParser",
    *,
    default: Optional[str] = None,
) -> None:
    if default is None:
        msg_default = "default: info, overwrites KTOOLBOX_LOGLEVEL environment"
    else:
        msg_default = f"default: {repr(default)}"
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["debug", "info", "warning", "error", "critical"],
        default=default,
        help=f"Set the logging level ({msg_default}). Set KTOOLBOX_ALL_LOGGERS to configure all loggers.",
    )


class ExtendedLogger(logging.Logger):
    """A wrapper around the logging.Logger class with an extended API.

    This class is a subclass of `logging.Logger`, and it delegates most functionality
    to the internal logger instance. It adds a few convenience methods but does not
    maintain any additional state. This means that, as long as the user interacts
    with the standard `Logger` API, there is no observable difference between using
    the `ExtendedLogger` or the wrapped `logging.Logger`.

    The purpose of this wrapper is to allow users to access the same logger instance
    whether they use `logging.getLogger(name)` or `ExtendedLogger(name)`. Since we
    cannot modify the behavior of `logging.getLogger()` directly to return a custom
    logging class, this wrapper allows us to offer extended functionality while
    maintaining compatibility with the standard logging system.

    Users of this `ExtendedLogger` can access the extended API while still interacting
    with the underlying logger instance, ensuring seamless integration with the standard
    logging framework.
    """

    def __init__(self, logger: Union[str, logging.Logger]):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        object.__setattr__(self, "wrapped_logger", logger)
        self.wrapped_logger: logging.Logger

    @staticmethod
    def unwrap(logger: Union[str, logging.Logger]) -> logging.Logger:
        if isinstance(logger, str):
            return logging.getLogger(logger)
        if isinstance(logger, ExtendedLogger):
            return logger.wrapped_logger
        return logger

    _EXTENDED_ATTRIBUTES = (
        "wrapped_logger",
        "error_and_exit",
        "unwrap",
        "__dir__",
        "__class__",
    )

    def __getattribute__(self, name: str) -> Any:
        # ExtendedLogger is-a logging.Logger, but it delegates most calls to
        # the wrapped-logger (which is also a logging.Logger).
        if name in ExtendedLogger._EXTENDED_ATTRIBUTES:
            return object.__getattribute__(self, name)
        logger = object.__getattribute__(self, "wrapped_logger")
        return logger.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ExtendedLogger._EXTENDED_ATTRIBUTES:
            raise AttributeError(f"{name} is read-only.")
        setattr(self.wrapped_logger, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ExtendedLogger._EXTENDED_ATTRIBUTES:
            raise AttributeError(f"{name} is read-only.")
        delattr(self.wrapped_logger, name)

    def __dir__(self) -> list[str]:
        logger = object.__getattribute__(self, "wrapped_logger")
        logger_attrs = set(dir(logger))
        logger_attrs.update(ExtendedLogger._EXTENDED_ATTRIBUTES)
        return sorted(logger_attrs)

    @typing.overload
    def error_and_exit(
        self,
        msg: str,
        *,
        exit_code: int = FATAL_EXIT_CODE,
        backtrace: bool = True,
        backtrace_with_exception: bool = True,
        die_on_error: Literal[False],
    ) -> None: ...

    @typing.overload
    def error_and_exit(
        self,
        msg: str,
        *,
        exit_code: int = FATAL_EXIT_CODE,
        backtrace: bool = True,
        backtrace_with_exception: bool = True,
        die_on_error: Literal[True] = True,
    ) -> typing.NoReturn: ...

    @typing.overload
    def error_and_exit(
        self,
        msg: str,
        *,
        exit_code: int = FATAL_EXIT_CODE,
        backtrace: bool = True,
        backtrace_with_exception: bool = True,
        die_on_error: bool = True,
    ) -> Union[None, typing.NoReturn]: ...

    def error_and_exit(
        self,
        msg: str,
        *,
        exit_code: int = FATAL_EXIT_CODE,
        backtrace: bool = True,
        backtrace_with_exception: bool = True,
        die_on_error: bool = True,
    ) -> Union[None, typing.NoReturn]:
        if not die_on_error:
            # Usually, error_and_exit() does what it says (it exists).
            #
            # However, if the caller also has a "die_on_error" variable, they
            # would need to do something like:
            #
            #    if die_on_error:
            #        logger.error_and_exit("error message")
            #    else:
            #        logger.error("error message")
            #
            # With the die_on_error argument, the caller can downgrade the
            # error_and_exit() to a plain error().
            self.error(msg)
            return None

        try:
            self.error(msg)
        except Exception:
            pass

        if backtrace:
            try:
                import traceback

                msg = "FATAL ERROR:\n"
                if backtrace_with_exception:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                    if exc_type is not None:
                        # There's an active exception, format its traceback
                        tb_str = "".join(
                            traceback.format_exception(exc_type, exc_value, exc_tb)
                        )
                        msg = f"FATAL ERROR (with exception):\n{tb_str}\nException came from:\n\n"

                msg += "".join(traceback.format_stack()[:-1])
                self.error(msg)
            except Exception:
                pass

        sys.exit(exit_code)


logger = ExtendedLogger(__name__)


class Cancellable:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._read_fd = -1
        self._write_fd = -1

    def __del__(self) -> None:
        with self._lock:
            fds = (self._read_fd, self._write_fd)
            self._read_fd = -1
            self._write_fd = -1
        for fd in fds:
            try:
                os.close(fd)
            except Exception:
                pass

    @staticmethod
    def is_cancelled(self: Optional["Cancellable"]) -> bool:
        return self is not None and self.cancelled

    def cancel(self) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            if self._read_fd != -1:
                os.write(self._write_fd, b"1")
        return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait_cancelled(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)

    def get_poll_fd(self) -> int:
        """
        Get a file descriptor to poll/select on. The descriptor will be
        ready to read when the cancellable is cancelled.

        Do not actually read from the descriptor. Only poll/select.
        """
        with self._lock:
            if self._read_fd == -1:
                self._read_fd, self._write_fd = os.pipe()
                if self._event.is_set():
                    os.write(self._write_fd, b"1")
            return self._read_fd


class FutureThread(typing.Generic[T1]):
    def __init__(
        self,
        func: typing.Callable[["FutureThread[T1]"], T1],
        *,
        start: bool = False,
        cancellable: Optional[Cancellable] = None,
        user_data: Any = None,
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
    ):
        self._lock = threading.Lock()
        self._func = func
        self.future: Future[T1] = Future()
        self._cancellable = cancellable
        self._cancellable_is_cancelled = False
        self._user_data = user_data
        self._is_started = False
        self._executor = executor
        self._thread: Optional[threading.Thread] = None
        if start:
            self.start()

    @property
    def cancellable(self) -> Cancellable:
        return self.get_cancellable()

    def get_cancellable(self) -> Cancellable:
        with self._lock:
            if self._cancellable is None:
                self._cancellable = Cancellable()
                if self._cancellable_is_cancelled:
                    self._cancellable.cancel()
            return self._cancellable

    def _cancel_with_lock(
        self,
    ) -> None:
        if self._cancellable is not None:
            self._cancellable.cancel()
        else:
            self._cancellable_is_cancelled = True

    def cancel(self) -> None:
        with self._lock:
            self._cancel_with_lock()

    @property
    def user_data(self) -> Any:
        with self._lock:
            return self._user_data

    def set_user_data(self, data: Any) -> Any:
        with self._lock:
            old = self._user_data
            self._user_data = data
            return old

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._is_started

    def start(self) -> bool:
        with self._lock:
            if self._is_started:
                return False
            self._is_started = True
            if self._executor is not None:
                self._executor.submit(self._run_task)
            else:
                self._thread = threading.Thread(target=self._run_task)
                self._thread.start()
        return True

    def ensure_started(self, *, start: bool = True) -> None:
        if start:
            self.start()

    def _run_task(self) -> None:
        try:
            result = self._func(self)
            self.future.set_result(result)
        except BaseException as e:
            self.future.set_exception(e)

    @typing.overload
    def result(
        self,
        *,
        timeout: typing.Literal[None] = None,
        cancel: bool = False,
    ) -> T1: ...

    @typing.overload
    def result(
        self,
        *,
        timeout: Optional[float],
        cancel: bool = False,
    ) -> Optional[T1]: ...

    def result(
        self,
        *,
        timeout: Optional[float] = None,
        cancel: bool = False,
    ) -> Optional[T1]:
        with self._lock:
            if not self._is_started:
                raise RuntimeError("thread is not yet started")
            if cancel:
                self._cancel_with_lock()
            thread = self._thread
        if self._executor is not None:
            try:
                return self.future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return None
        else:
            assert thread is not None
            thread.join(timeout=timeout)
            if thread.is_alive():
                return None
            return self.future.result()

    def poll(
        self,
        *,
        timeout: float = 0.0,
        cancel: bool = False,
    ) -> Optional[T1]:
        return self.result(
            timeout=timeout,
            cancel=cancel,
        )

    def join(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        self.result(timeout=timeout)

    def join_and_result(self, *, cancel: bool = False) -> T1:
        return self.result(cancel=cancel)


_thread_list: list[Union[threading.Thread, FutureThread[Any]]] = []


def thread_list_get() -> list[Union[threading.Thread, FutureThread[Any]]]:
    with common_lock:
        return list(_thread_list)


def thread_list_add(
    self: Union[threading.Thread, FutureThread[Any]],
    *,
    start: bool = False,
) -> None:
    with common_lock:
        _thread_list.append(self)
        if start:
            if isinstance(self, threading.Thread):
                try:
                    self.start()
                except RuntimeError:
                    pass
            else:
                self.start()


def thread_list_cancel(
    *,
    threads: Optional[Iterable[Union[threading.Thread, FutureThread[Any]]]] = None,
) -> None:
    if threads is None:
        threads = thread_list_get()
    else:
        threads = iter_eval_now(threads)
    for th in threads:
        if isinstance(th, FutureThread):
            th.cancel()


def thread_list_join_all(
    *,
    cancel: bool = True,
    threads: Optional[Iterable[Union[threading.Thread, FutureThread[Any]]]] = None,
) -> None:
    if threads is not None:
        threads = iter_eval_now(threads)
    lst_idx = 0
    while True:
        if threads is not None:
            if lst_idx >= len(threads):
                return
            th = threads[lst_idx]
            lst_idx += 1
        else:
            # There is a difference between passing a "threads" argument and
            # None.  Here we repeat the loop unil the thread list is empty. If
            # a thread argument is provided, the list is evaluated once at the
            # beginning.  That makes a difference if the list gets modified
            # while calling join-all.
            with common_lock:
                if not _thread_list:
                    return
                th = _thread_list.pop(0)
        if isinstance(th, FutureThread):
            th.result(cancel=cancel)
        else:
            th.join()


@functools.cache
def get_current_host() -> str:
    chost = getenv_config("KTOOLBOX_CURRENT_HOST")
    if chost:
        return chost

    from . import host

    res = host.local.run(
        ("hostname", "-f"),
        log_level_fail=logging.ERROR,
    )
    if res.success and (c := res.out.strip()):
        return c
    raise RuntimeError(f"Failure detecting current hostname: {res}")


def argparse_regex_type(value: str) -> re.Pattern[str]:
    try:
        return re.compile(value)
    except re.error as e:
        import argparse

        raise argparse.ArgumentTypeError(f"Invalid regex pattern: {e}")


# Error codes from <sysexits.h>
EX_SOFTWARE = 70
EX_CONFIG = 78


def run_main(
    main_fcn: Union[
        typing.Callable[[], None],
        typing.Callable[[], int],
        typing.Callable[[], Optional[int]],
    ],
    *,
    error_code: int = EX_SOFTWARE,
    logger: Optional[logging.Logger] = logger,
    error_code_keyboard_interrupt: Optional[int] = None,
    cleanup: Optional[typing.Callable[[], None]] = None,
) -> typing.NoReturn:
    try:
        got_exception: Optional[BaseException] = None
        try:
            exit_code = main_fcn()
        except BaseException as e:
            got_exception = e
            raise
        finally:
            if cleanup is not None:
                if got_exception and logger is not None:
                    logger.info(f"Got exception {got_exception}. Run cleanup first")
                cleanup()
    except Exception:
        import traceback

        if logger is None:
            traceback.print_exc()
        else:
            logger.error(f"FATAL ERROR:\n{traceback.format_exc()}")

        exit_code = error_code
    except KeyboardInterrupt:
        if error_code_keyboard_interrupt is None:
            raise
        exit_code = error_code_keyboard_interrupt

    if exit_code is None:
        exit_code = 0
    sys.exit(exit_code)


def format_duration(seconds: float) -> str:
    if seconds < 0:
        s_sign = "-"
        seconds = -seconds
    else:
        s_sign = ""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    secs = round(secs, 4)
    s = s_sign
    if days != 0:
        s += f"{days}d-"
    s += f"{hours:02}:"
    s += f"{minutes:02}:"
    s += f"{secs:07.4f}"
    return s


def getenv_config(name: str) -> Optional[str]:
    """
    Return the value of a user-configurable environment variable.

    This wraps os.getenv() for the purpose of marking environment variables
    that are intended for user configuration. By using this function, it's easy
    to locate all such variables via grep or static analysis.
    """
    return os.getenv(name)


def time_monotonic(now: Optional[float]) -> float:
    if now is None:
        return time.monotonic()
    return float(now)


class NextExpiry:
    @staticmethod
    def _next(
        expiry_1: Optional[float],
        expiry_2: Optional[float] = None,
    ) -> Optional[float]:
        if expiry_1 is None:
            if expiry_2 is None:
                return None
            return float(expiry_2)
        if expiry_2 is None:
            return float(expiry_1)
        return min(float(expiry_1), float(expiry_2))

    def __init__(self, expiry: Optional[Union[float, "NextExpiry"]] = None) -> None:
        self._expiry: Optional[float]
        if isinstance(expiry, NextExpiry):
            expiry = expiry._expiry
        self._expiry = NextExpiry._next(expiry)

    def reset(self, expiry: Optional[float] = None) -> None:
        self._expiry = NextExpiry._next(expiry)

    def update(
        self,
        *,
        now: Optional[float] = None,
        timeout: Optional[float] = None,
        expiry: Optional[float] = None,
    ) -> None:
        if timeout is None and expiry is None:
            return
        if timeout is not None:
            expiry = NextExpiry._next(
                expiry,
                time_monotonic(now) + float(timeout),
            )
        self._expiry = NextExpiry._next(self._expiry, expiry)

    @staticmethod
    def up(
        self: Optional["NextExpiry"],
        *,
        now: Optional[float] = None,
        timeout: Optional[float] = None,
        expiry: Optional[float] = None,
    ) -> None:
        if self is None:
            return
        self.update(now=now, timeout=timeout, expiry=expiry)

    def expires_at(
        self,
        *,
        now: Optional[float] = None,
        expiry: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[float]:
        next_expiry = NextExpiry(self)
        next_expiry.update(now=now, expiry=expiry, timeout=timeout)
        return next_expiry._expiry

    def expires_in(
        self,
        *,
        now: Optional[float] = None,
        expiry: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[float]:
        now = time_monotonic(now)
        expiry = self.expires_at(now=now, expiry=expiry, timeout=timeout)
        if expiry is None:
            return None
        return expiry - now
