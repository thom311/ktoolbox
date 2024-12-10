import jinja2
import pathlib
import typing

from typing import Any
from typing import Optional
from typing import Union

from . import common


def render_data(
    contents: str,
    kwargs: dict[str, Any],
) -> str:
    template = jinja2.Template(contents)
    rendered = template.render(**kwargs)
    return rendered


def render_file(
    in_file: Union[str, pathlib.Path, typing.IO[str]],
    kwargs: dict[str, Any],
    *,
    out_file: Optional[Union[str, pathlib.Path, typing.IO[str]]] = None,
) -> str:
    with common.use_or_open(in_file) as inFile:
        contents = inFile.read()
    rendered = render_data(contents, kwargs)
    if out_file is not None:
        with common.use_or_open(out_file, mode="w") as outFile:
            outFile.write(rendered)
    return rendered
