import io
import pathlib
import typing
import yaml

from ktoolbox import kyaml


def test_yamldumper(tmp_path: pathlib.Path) -> None:
    data = {
        "foo": [
            {"id": 1},
            {"id": 2},
        ]
    }

    assert (
        yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
        )
        == """foo:
- id: 1
- id: 2
"""
    )

    def _check(data: typing.Any, result: str) -> None:
        assert (
            yaml.dump(
                data,
                default_flow_style=False,
                sort_keys=False,
                Dumper=kyaml.YamlDumper,
            )
            == result
        )

        assert kyaml.dumps(data) == result

        buffer = io.StringIO()
        kyaml.dump(data, buffer)
        assert buffer.getvalue() == result

        tmp_file = tmp_path / "file1.yaml"
        kyaml.dump(data, tmp_file)
        with open(tmp_file) as f:
            assert f.read() == result

        tmp_file_s = str(tmp_path / "file2.yaml")
        kyaml.dump(data, tmp_file_s)
        with open(tmp_file_s) as f:
            assert f.read() == result

    _check(
        data,
        """foo:
  - id: 1
  - id: 2
""",
    )
