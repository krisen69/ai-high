from pathlib import Path


def test_real_pydantic_is_used() -> None:
    import pydantic

    module_path = Path(pydantic.__file__).resolve()
    assert module_path.name != "pydantic.py"
    assert "vod-highlight-storyline" not in str(module_path)
