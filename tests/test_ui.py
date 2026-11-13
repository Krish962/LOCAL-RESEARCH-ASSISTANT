def test_entrypoint_import_does_not_launch_or_create_backend():
    import app,enhanced_app
    import assistant.ui as ui
    assert not hasattr(app,'demo') and not hasattr(enhanced_app,'demo')
    assert ui._service is None


def test_optional_year_parser_rejects_ambiguous_or_invalid_input():
    import pytest
    from assistant.ui import optional_year
    assert optional_year('') is None and optional_year(None) is None
    assert optional_year('2023') == 2023
    with pytest.raises(ValueError):
        optional_year('2023.5')
    with pytest.raises(ValueError):
        optional_year('0')
