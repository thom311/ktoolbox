import io
import pathlib

from typing import Any

from ktoolbox import kjinja2

import tstutil


def test_render_data(tmp_path: pathlib.Path) -> None:
    def _r(contents: str, **kwargs: Any) -> str:
        filename = tmp_path / "outfile1"
        out_file = tstutil.rnd_select(
            None,
            filename,
            str(filename),
            io.StringIO(),
        )

        rendered = kjinja2.render_data(contents, kwargs, out_file=out_file)

        if out_file is not None:
            if isinstance(out_file, io.StringIO):
                out_file.seek(0)
                re_read = out_file.read()
            else:
                with open(out_file) as f:
                    re_read = f.read()
            assert rendered == re_read
        return rendered

    assert _r("", a="1") == ""
    assert _r("val: {{a}}", a=1) == "val: 1"
    assert _r("val: {{a}}", a="1") == "val: 1"
    assert _r("val: {{a}}", a="a") == "val: a"
    assert _r("val: {{a}}", a="a b") == "val: a b"
    assert _r("val: {{a|tojson}}", a=1) == "val: 1"
    assert _r("val: {{a|tojson}}", a="1") == 'val: "1"'
    assert _r("val: {{a|tojson}}", a="a") == 'val: "a"'
    assert _r("val: {{a|tojson}}", a="a b") == 'val: "a b"'

    assert (
        kjinja2.render_data("foo[{{x1}}][{{x2}}]", kwargs={"x1": "1"}, x2=2)
        == "foo[1][2]"
    )
    assert (
        kjinja2.render_data("foo[{{x1}}][{{x2}}]", {"x1": "1"}, x1="b", x2=2)
        == "foo[b][2]"
    )
    assert kjinja2.render_data("foo{{xx}}{{x1}}", x="a", x1=0) == "foo0"


def test_render_file() -> None:
    in_buffer = io.StringIO("val: {{a}}")
    out_buffer = io.StringIO()
    y1 = kjinja2.render_file(in_buffer, {"a": 1}, out_file=out_buffer)
    assert y1 == "val: 1"
    assert out_buffer.getvalue() == y1
