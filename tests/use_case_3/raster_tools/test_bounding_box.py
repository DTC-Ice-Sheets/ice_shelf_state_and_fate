"""Test the BoundingBox class."""
import pytest

from use_case_3.raster_tools.bounding_box import BoundingBox


def test_bounding_box():
    x_min = 1
    x_max = 2
    y_min = 10
    y_max = 12

    bounding_box = BoundingBox(x_min, x_max, y_min, y_max)

    assert bounding_box.__str__() == "min_x=1, max_x=2, min_y=10, max_y=12"


def test_bounding_box_raises_error():
    x_min = 2
    x_max = 1
    y_min = 10
    y_max = 12

    with pytest.raises(ValueError, match="cannot be greater"):
        BoundingBox(x_min, x_max, y_min, y_max)


def test_bounding_box_intersect():
    bounding_box_1 = BoundingBox(0, 2, 4, 6)
    bounding_box_2 = BoundingBox(1, 2, 4, 5)
    bounding_box_3 = BoundingBox(-1, 3, 3, 7)
    bounding_box_4 = BoundingBox(0, 2, 7, 8)
    bounding_box_5 = BoundingBox(-1, 3, 5, 7)
    bounding_box_6 = BoundingBox(2, 3, 4, 6)

    # Simple overlaps
    assert bounding_box_1.intersects(bounding_box_2)
    assert bounding_box_2.intersects(bounding_box_1)
    assert bounding_box_1.intersects(bounding_box_3)
    assert bounding_box_3.intersects(bounding_box_1)
    # Non overlap
    assert not bounding_box_1.intersects(bounding_box_4)
    # Odd case
    assert bounding_box_1.intersects(bounding_box_5)
    # Touching
    assert not bounding_box_1.intersects(bounding_box_6)


def test_bounding_box_intersect_3d():
    bounding_box_1 = BoundingBox(0, 2, 4, 6, 0, 2)
    bounding_box_2 = BoundingBox(1, 2, 4, 5, 1, 3)
    bounding_box_3 = BoundingBox(1, 2, 4, 5, 2, 4)

    # Simple overlaps
    assert bounding_box_1.intersects(bounding_box_2)
    assert not bounding_box_1.intersects(bounding_box_3)


def test_bounding_box_z_extent_defined():
    assert not (BoundingBox(min_x=4, max_x=6).z_extent_defined())
    assert BoundingBox(min_z=4, max_z=6).z_extent_defined()


def test_bounding_box_str():
    bounding_box = BoundingBox(0, 2, 4, 6)
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6"


def test_bounding_boxs_str_t():
    bounding_box = BoundingBox(0, 2, 4, 6, min_t=0, max_t=2)
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6, min_t=0, max_t=2"


def test_bounding_boxs_str_z():
    bounding_box = BoundingBox(0, 2, 4, 6, min_z=0, max_z=2)
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6, min_z=0, max_z=2"


def test_bounding_box_from_dict():
    bounding_box = BoundingBox.from_dict({"min_x": 0, "max_x": 2, "min_y": 4, "max_y": 6})
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6"
    bounding_box = BoundingBox.from_dict({"min_x": 0, "max_x": 2, "min_y": 4, "max_y": 6, "min_t": 0, "max_t": 2})
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6, min_t=0, max_t=2"
    bounding_box = BoundingBox.from_dict({"min_x": 0, "max_x": 2, "min_y": 4, "max_y": 6, "min_z": 0, "max_z": 2})
    assert str(bounding_box) == "min_x=0, max_x=2, min_y=4, max_y=6, min_z=0, max_z=2"


def test_bounding_box_dict_round_trip():
    bounding_box = BoundingBox(0, 2, 4, 6, min_z=0, max_z=2)
    print(BoundingBox.from_dict(bounding_box.as_dict()))
    print(str(bounding_box))
    assert str(bounding_box) == str(BoundingBox.from_dict(bounding_box.as_dict()))
