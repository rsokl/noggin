def test_version():
    import liveplot

    assert isinstance(liveplot.__version__, str)
    assert liveplot.__version__
    assert "unknown" not in liveplot.__version__
