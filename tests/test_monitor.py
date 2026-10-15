import pytest

from app.monitor import parse_polygon


def test_parse_polygon_accepts_normalized_points():
    assert parse_polygon("0,0 1,0 1,1") == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_parse_polygon_rejects_invalid_coordinate():
    with pytest.raises(ValueError):
        parse_polygon("0,0 2,0 1,1")


def test_parse_polygon_requires_three_points():
    with pytest.raises(ValueError):
        parse_polygon("0,0 1,0")
