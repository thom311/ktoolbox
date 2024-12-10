import pathlib
import typing
import yaml

from typing import Any
from typing import Union

from . import common


class YamlDumper(yaml.SafeDumper):
    """
    Dumper for yaml.dump(Dumper=) which indents YAML like `yq -P` would do

    https://github.com/yaml/pyyaml/issues/234#issuecomment-765894586
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        fcn: typing.Callable[[bool, bool], None] = super().increase_indent
        fcn(flow, False)


def dump(
    arg: Any,
    file: Union[str, pathlib.Path, typing.IO[str]],
) -> None:
    with common.use_or_open(file, mode="w") as f:
        yaml.dump(
            arg,
            stream=f,
            default_flow_style=False,
            sort_keys=False,
            Dumper=YamlDumper,
        )


def dumps(arg: Any) -> str:
    s: str = yaml.dump(
        arg,
        default_flow_style=False,
        sort_keys=False,
        Dumper=YamlDumper,
    )
    return s
