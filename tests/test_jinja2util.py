import io

from typing import Any

from ktoolbox import jinja2util


def test_render() -> None:
    def _r(contents: str, **kwargs: Any) -> str:
        return jinja2util.render_data(contents, kwargs)

    assert _r("", a="1") == ""
    assert _r("val: {{a}}", a=1) == "val: 1"
    assert _r("val: {{a}}", a="1") == "val: 1"
    assert _r("val: {{a}}", a="a") == "val: a"
    assert _r("val: {{a}}", a="a b") == "val: a b"
    assert _r("val: {{a|tojson}}", a=1) == "val: 1"
    assert _r("val: {{a|tojson}}", a="1") == 'val: "1"'
    assert _r("val: {{a|tojson}}", a="a") == 'val: "a"'
    assert _r("val: {{a|tojson}}", a="a b") == 'val: "a b"'


def test_render_file() -> None:
    in_buffer = io.StringIO("val: {{a}}")
    out_buffer = io.StringIO()
    jinja2util.render_file(in_buffer, out_buffer, {"a": 1})
    assert out_buffer.getvalue() == "val: 1"
