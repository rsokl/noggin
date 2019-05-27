def test_version():
    import noggin

    assert isinstance(noggin.__version__, str)
    assert noggin.__version__
    assert "unknown" not in noggin.__version__
