"""Test the RasterNp class."""
import os

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from numpy.typing import DTypeLike, NDArray

from use_case_3.raster_tools.bounding_box import BoundingBox
from use_case_3.raster_tools.raster_np import RasterNp

PROJ4_CODE_FIXTURE = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True"


@pytest.fixture
def np_arr_fixture() -> NDArray:
    return np.array([[1, 2], [3, 4], [6, 6]])


@pytest.fixture
def bounding_box_fixture() -> BoundingBox:
    x_min = 0
    x_max = 200
    y_min = 400
    y_max = 700
    return BoundingBox(x_min, x_max, y_min, y_max)


@pytest.fixture
def y_affine_negative_fixture(np_arr_fixture: NDArray, bounding_box_fixture: BoundingBox):
    resolution = 100
    return RasterNp(
        np_arr_fixture, bounding_box_fixture, resolution, y_affine_negative=True, proj4_code=PROJ4_CODE_FIXTURE)


@pytest.fixture
def y_affine_positive_fixture(np_arr_fixture: NDArray, bounding_box_fixture: BoundingBox):
    resolution = 100
    return RasterNp(
        np_arr_fixture, bounding_box_fixture, resolution, y_affine_negative=False, proj4_code=PROJ4_CODE_FIXTURE)


@pytest.fixture
def dummy_raster():
    np_arr_nan = np.array([[np.nan, 2], [3, 4], [6, 6]])
    raster_np = RasterNp(
        np_arr_nan, BoundingBox(0, 200, 400, 700), 100, y_affine_negative=True, proj4_code=PROJ4_CODE_FIXTURE)
    return raster_np


@pytest.fixture
def test_inputs_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_inputs")


def assert_save_and_load_tiff(save_path: str, raster_np: RasterNp, data_type: DTypeLike = None):
    if data_type is None:
        raster_np.save_as_tiff(save_path)
        raster_np_test = raster_np
    else:
        raster_np.save_as_tiff(save_path, data_type=data_type)
        raster_np_test = raster_np.astype(data_type)

    assert os.path.exists(save_path)

    raster_np_loaded = RasterNp.load_from_tiff(save_path)
    np.testing.assert_array_equal(raster_np_test, raster_np_loaded)

    assert raster_np.resolution == raster_np_loaded.resolution
    assert raster_np.bounding_box.min_x == raster_np_loaded.bounding_box.min_x
    assert raster_np.bounding_box.max_x == raster_np_loaded.bounding_box.max_x
    assert raster_np.bounding_box.min_y == raster_np_loaded.bounding_box.min_y
    assert raster_np.bounding_box.max_y == raster_np_loaded.bounding_box.max_y
    #  note that the two proj4 codes below are equivalent,
    # e.g. see https://www.spatialreference.org/ref/epsg/wgs-84-nsidc-sea-ice-polar-stereographic-north/
    assert (len({raster_np.proj4_code, raster_np_loaded.proj4_code}
            - {"+init=epsg:3413",
               "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True"}) == 0
            or (raster_np.proj4_code == raster_np_loaded.proj4_code))


def test_raster_np_init_y_affine_positive(bounding_box_fixture: BoundingBox, y_affine_positive_fixture: RasterNp):

    assert y_affine_positive_fixture.resolution == 100
    assert y_affine_positive_fixture.bounding_box.min_x == bounding_box_fixture.min_x
    assert y_affine_positive_fixture.bounding_box.max_x == bounding_box_fixture.max_x
    assert y_affine_positive_fixture.bounding_box.min_y == bounding_box_fixture.min_y
    assert y_affine_positive_fixture.bounding_box.max_y == bounding_box_fixture.max_y
    assert not y_affine_positive_fixture.y_affine_negative
    assert y_affine_positive_fixture.proj4_code == PROJ4_CODE_FIXTURE


def test_raster_np_init_y_affine_negative(y_affine_negative_fixture: RasterNp):
    assert y_affine_negative_fixture.y_affine_negative


def test_raster_np_init_odd_extent(np_arr_fixture: NDArray):
    resolution = 100
    bounding_box = BoundingBox(20, 220, 420, 720)
    raster_arr = RasterNp(np_arr_fixture, bounding_box, resolution, False, PROJ4_CODE_FIXTURE)
    assert raster_arr.yx_from_rows_cols(0, 0) == (470, 70)


def test_raster_np_yx_from_cols_y_affine_positive(y_affine_positive_fixture: RasterNp):
    assert y_affine_positive_fixture.yx_from_rows_cols(0, 0) == (450.0, 50.0)
    assert y_affine_positive_fixture.yx_from_rows_cols(2, 1) == (650.0, 150)


def test_raster_np_yx_from_cols_y_affine_negative(y_affine_negative_fixture: RasterNp):
    assert y_affine_negative_fixture.yx_from_rows_cols(0, 0) == (650.0, 50.0)
    assert y_affine_negative_fixture.yx_from_rows_cols(2, 1) == (450.0, 150)


def test_raster_np_rows_cols_from_yx_y_affine_positive(y_affine_positive_fixture: RasterNp):
    assert y_affine_positive_fixture.rows_cols_from_yx(475, 5) == (0, 0)
    assert y_affine_positive_fixture.rows_cols_from_yx(699, 199) == (2, 1)


def test_raster_np_rows_cols_from_yx_y_affine_negative(y_affine_negative_fixture: RasterNp):
    assert y_affine_negative_fixture.rows_cols_from_yx(401, 1) == (2, 0)
    assert y_affine_negative_fixture.rows_cols_from_yx(510, 110) == (1, 1)
    assert y_affine_negative_fixture.rows_cols_from_yx(699, 199) == (0, 1)

    rows, cols = y_affine_negative_fixture.rows_cols_from_yx((401, 699), (199, 1))
    np.testing.assert_array_equal(rows, (2, 0))
    np.testing.assert_array_equal(cols, (1, 0))


def test_load_tiff_nan_no_values_ints(tmpdir: str, y_affine_negative_fixture: RasterNp):
    file_path = os.path.join(tmpdir, "novaluetest.tiff")

    y_affine_negative_fixture.save_as_tiff(file_path, data_type="uint8", no_data=1)

    loaded_raster = y_affine_negative_fixture.load_from_tiff(file_path)
    assert np.isnan(loaded_raster[0, 0])
    assert ~np.isnan(loaded_raster[0, 1])


def test_load_tiff_nan_no_values_doubles_extents(tmpdir: str, y_affine_negative_fixture: RasterNp):
    file_path = os.path.join(tmpdir, "novaluetest_dbl.tiff")

    y_affine_negative_fixture.save_as_tiff(file_path, no_data=6)
    loaded_raster = y_affine_negative_fixture.load_from_tiff(
        file_path, bounding_box=BoundingBox(0.0, 200.0, 400.0, 600.0))
    assert np.isnan(loaded_raster[1, 1])
    assert ~np.isnan(loaded_raster[0, 0])


def test_raster_numpy_save_nan(tmpdir: str, dummy_raster: RasterNp):
    save_path = os.path.join(tmpdir, "nan_test.tiff")
    assert_save_and_load_tiff(save_path, dummy_raster)


def test_raster_numpy_save_int(tmpdir: str, dummy_raster: RasterNp):
    save_path = os.path.join(tmpdir, "nan_test.tiff")
    assert_save_and_load_tiff(save_path, dummy_raster, data_type="int32")


def test_raster_numpy_save_and_load_y_affine_positive(tmpdir: str, y_affine_positive_fixture: RasterNp):
    save_path = os.path.join(tmpdir, "affine_positive.tiff")
    assert_save_and_load_tiff(save_path, y_affine_positive_fixture)


def test_raster_numpy_save_and_load_y_affine_negative(tmpdir: str, y_affine_negative_fixture: RasterNp):
    save_path = os.path.join(tmpdir, "affine_negative.tiff")
    assert_save_and_load_tiff(save_path, y_affine_negative_fixture)


def test_raster_numpy_load_sub_extent(test_inputs_path: str):

    test_file = os.path.join(test_inputs_path, "rasters", "4x4.tiff")
    # Actual extent = min_x=50.0, max_x=450.0, min_y=1050.0, max_y=1450.0
    sub_extent = BoundingBox(min_x=150, max_x=450.0, min_y=1150.0, max_y=1350.0)

    arr = RasterNp.load_from_tiff(test_file, bounding_box=sub_extent)

    np.testing.assert_array_equal(arr, np.array([[5, 6, 7], [9, 10, 11]]))
    assert arr.bounding_box.min_x == 150
    assert arr.bounding_box.min_y == 1150


def test_raster_numpy_load_sub_extent_sub_pixels(test_inputs_path: str):

    test_file = os.path.join(test_inputs_path, "rasters", "4x4.tiff")
    # Actual extent = min_x=50.0, max_x=450.0, min_y=1050.0, max_y=1450.0
    sub_extent = BoundingBox(min_x=210, max_x=310.0, min_y=1210.0, max_y=1340.0)

    arr = RasterNp.load_from_tiff(test_file, bounding_box=sub_extent)
    np.testing.assert_array_equal(arr, np.array([[5, 6], [9, 10]]))
    assert arr.bounding_box.min_x == 150
    assert arr.bounding_box.min_y == 1150


def test_raster_numpy_load_with_extent_greater_than_image(test_inputs_path: str):

    test_file = os.path.join(test_inputs_path, "rasters", "4x4.tiff")
    # Actual extent = min_x=50.0, max_x=450.0, min_y=1050.0, max_y=1450.0
    sub_extent = BoundingBox(min_x=-100, max_x=500, min_y=0, max_y=2000)

    arr = RasterNp.load_from_tiff(test_file, bounding_box=sub_extent)

    np.testing.assert_array_equal(arr, np.arange(16).reshape((4, 4)))
    assert arr.bounding_box.min_x == 50
    assert arr.bounding_box.min_y == 1050


def test_yx_gridpoints(y_affine_negative_fixture: RasterNp):
    y, x = y_affine_negative_fixture.yx_gridpoints()

    np.testing.assert_array_equal(np.array([50, 150]), x)
    np.testing.assert_array_equal(np.array([450, 550, 650]), y)


def test_at_yx(y_affine_negative_fixture: RasterNp):
    values = y_affine_negative_fixture.at_yx([550, 650], [50, 150])

    assert values[0] == 3
    assert values[1] == 2


def test_new_extent_copy_simple(y_affine_negative_fixture: RasterNp):
    extent = BoundingBox(100, 200, 500, 700)

    new_raster_np = y_affine_negative_fixture.new_extent_copy(extent)
    np.testing.assert_array_equal(new_raster_np, np.array([[2], [4]]))


def test_new_extent_copy_affine_positive(y_affine_positive_fixture: RasterNp):
    extent = BoundingBox(100, 200, 500, 700)

    new_raster_np = y_affine_positive_fixture.new_extent_copy(extent)
    np.testing.assert_array_equal(new_raster_np, np.array([[4], [6]]))


def test_new_extent_copy_2d(y_affine_negative_fixture: RasterNp):
    extent = BoundingBox(0, 200, 500, 700)

    new_raster_np = y_affine_negative_fixture.new_extent_copy(extent)
    np.testing.assert_array_equal(new_raster_np, np.array([[1, 2], [3, 4]]))

    # Make sure a copy is created and updating the original object doesn't update the new copy
    new_raster_np[0, 0] = 10
    assert y_affine_negative_fixture[0, 0] != 10

    assert new_raster_np.bounding_box.min_x == 0
    assert new_raster_np.bounding_box.max_x == 200
    assert new_raster_np.bounding_box.min_y == 500
    assert new_raster_np.bounding_box.max_y == 700


def test_new_extent_copy_extent_larger_than_array(y_affine_negative_fixture: RasterNp, np_arr_fixture: NDArray):
    extent = BoundingBox(-1000, 5000, -10000, 10000)

    arr_test = y_affine_negative_fixture.new_extent_copy(extent)
    np.testing.assert_array_equal(arr_test, np_arr_fixture)


def test_new_extent_copy_buffer_corner_partial_overlap(y_affine_negative_fixture: RasterNp):
    bounding_box = BoundingBox(100, 300, 500, 800)
    arr_test = y_affine_negative_fixture.new_extent_copy(bounding_box, fill_to_extent=True)
    assert arr_test.bounding_box.min_x == 100
    assert arr_test.bounding_box.max_x == 300
    assert arr_test.bounding_box.min_y == 500
    assert arr_test.bounding_box.max_y == 800
    np.testing.assert_array_equal(arr_test[0], np.array([np.nan, np.nan]))
    np.testing.assert_array_equal(arr_test[2], np.array([4, np.nan]))


def test_new_extent_copy_buffer_corner_partial_overlap_no_fill(y_affine_negative_fixture: RasterNp):
    bounding_box = BoundingBox(100, 300, 500, 800)
    arr_test = y_affine_negative_fixture.new_extent_copy(bounding_box, fill_to_extent=False)
    assert arr_test.bounding_box.min_x == 100
    assert arr_test.bounding_box.max_x == 200
    assert arr_test.bounding_box.min_y == 500
    assert arr_test.bounding_box.max_y == 700
    np.testing.assert_array_equal(arr_test, np.array([[2], [4]]))


def test_new_extent_copy_buffer_extent_larger_than_array(y_affine_negative_fixture: RasterNp):
    bounding_box = BoundingBox(-200, 500, 100, 900)
    arr_test = y_affine_negative_fixture.new_extent_copy(bounding_box, fill_to_extent=True)
    assert arr_test.bounding_box.min_x == -200
    assert arr_test.bounding_box.max_x == 500
    assert arr_test.bounding_box.min_y == 100
    assert arr_test.bounding_box.max_y == 900
    np.testing.assert_array_equal(arr_test[0], np.ones(7) * np.nan)
    np.testing.assert_array_equal(arr_test[2], np.array([np.nan, np.nan, 1, 2, np.nan, np.nan, np.nan]))


def test_get_resolution_rounded_extent(y_affine_negative_fixture: RasterNp):
    extent = BoundingBox(-250, 500, 70, 911)
    round_extent = y_affine_negative_fixture._get_resolution_rounded_bounding_box(extent)
    assert round_extent.min_x == -300
    assert round_extent.max_x == 500
    assert round_extent.min_y == 0
    assert round_extent.max_y == 1000


def test_get_resolution_rounded_extent_odd_extent(y_affine_negative_fixture: RasterNp):
    y_affine_negative_fixture.bounding_box = BoundingBox(
        y_affine_negative_fixture.bounding_box.min_x - 50,
        y_affine_negative_fixture.bounding_box.max_x - 50,
        y_affine_negative_fixture.bounding_box.min_y - 50,
        y_affine_negative_fixture.bounding_box.max_y - 50)

    extent = BoundingBox(-250, 500, 70, 911)
    round_extent = y_affine_negative_fixture._get_resolution_rounded_bounding_box(extent)
    assert round_extent.min_x == -250
    assert round_extent.max_x == 550
    assert round_extent.min_y == 50
    assert round_extent.max_y == 950


def test_extent_intersects_tiff(test_inputs_path: str):

    mask_file = os.path.join(test_inputs_path, "rasters", "small_mask.tiff")

    # Inside Tiff
    extent = BoundingBox(-1568906, -1563418, -338635, -334977)
    assert RasterNp.bounding_box_intersects_tiff(mask_file, extent)

    # Outside Tiff
    extent = BoundingBox(-1568906 + 100000, -1568906 + 100001, -334977 + 100000, -334977 + 100001)
    assert not RasterNp.bounding_box_intersects_tiff(mask_file, extent)


def test_raster_np_get_centre_points(y_affine_negative_fixture: RasterNp):
    x, y, values = y_affine_negative_fixture.get_centre_points()

    np.testing.assert_array_equal(np.array([50, 150]), x)
    np.testing.assert_array_equal(np.array([450, 550, 650]), y)

    arr_expected = np.array([[6, 3, 1], [6, 4, 2]])
    np.testing.assert_array_equal(arr_expected, values)


def test_raster_np_mask_from_file(test_inputs_path: str, y_affine_negative_fixture: RasterNp):

    mask_path = os.path.join(test_inputs_path, "rasters", "simple_mask.tiff")

    y_affine_negative_fixture.mask_from_file(mask_path, outside_mask_return_value=0)

    assert y_affine_negative_fixture[0, 1] == 2
    assert y_affine_negative_fixture[0, 0] == 0


def test_raster_np_mask_from_file_include_mask_value(test_inputs_path: str, y_affine_negative_fixture: RasterNp):

    mask_path = os.path.join(test_inputs_path, "rasters", "simple_mask.tiff")
    y_affine_negative_fixture.mask_from_file(mask_path, include_mask_value=0, outside_mask_return_value=0)

    assert y_affine_negative_fixture[0, 1] == 0
    assert y_affine_negative_fixture[0, 0] == 1


def test_raster_np_mask_from_file_include_mask_value_affine_positive(test_inputs_path: str,
                                                                     y_affine_positive_fixture: RasterNp):

    mask_path = os.path.join(test_inputs_path, "rasters", "simple_mask.tiff")
    y_affine_positive_fixture.mask_from_file(mask_path, outside_mask_return_value=100)

    assert y_affine_positive_fixture[2, 1] == 6
    assert y_affine_positive_fixture[0, 1] == 100


def test_convert_to_dataframe(y_affine_negative_fixture: RasterNp):
    df1 = y_affine_negative_fixture.convert_to_dataframe()
    df1a = pd.DataFrame({"x": [50.0, 50.0, 50.0, 150.0, 150.0, 150.0], "y": [
                        450.0, 550.0, 650.0, 450.0, 550.0, 650.0], "value": [6, 3, 1, 6, 4, 2]})

    df2 = y_affine_negative_fixture.convert_to_dataframe("goose")
    df2a = pd.DataFrame({"x": [50.0, 50.0, 50.0, 150.0, 150.0, 150.0], "y": [
                        450.0, 550.0, 650.0, 450.0, 550.0, 650.0], "goose": [6, 3, 1, 6, 4, 2]})

    assert len(df1) == 6
    pdt.assert_frame_equal(df1, df1a)
    pdt.assert_frame_equal(df2, df2a)


def test_bounding_box_partial_intersect_with_raster(test_inputs_path: str):
    # Test for an edge case where, in some cases, rounding the window led the bounding box
    # to expand beyond the complete raster bounding box and not match the read-in raster's shape.
    test_file = os.path.join(test_inputs_path, "rasters", "svalbard_mask_sub_extent_for_window_edge_case.tif")

    sub_extent = BoundingBox(min_x=976290.688, min_y=-1902572.555, max_x=2671651.758, max_y=-478302.955)
    raster = RasterNp.load_from_tiff(test_file, bounding_box=sub_extent)

    expected_raster = RasterNp.load_from_tiff(
        os.path.join(test_inputs_path, "rasters", "svalbard_sub_extent.tif"))

    np.testing.assert_array_equal(raster.data, expected_raster.data)
    assert abs(raster.bounding_box.min_x - expected_raster.bounding_box.min_x) < 1e-4
    assert abs(raster.bounding_box.max_x - expected_raster.bounding_box.max_x) < 1e-4
    assert abs(raster.bounding_box.min_y - expected_raster.bounding_box.min_y) < 1e-4
    assert abs(raster.bounding_box.max_y - expected_raster.bounding_box.max_y) < 1e-4
    assert raster.transform == expected_raster.transform


def np_arr_() -> NDArray:
    return np.array([[1, 2], [3, 4], [6, 6]])


def bounding_box_() -> BoundingBox:
    x_min = 0
    x_max = 200
    y_min = 400
    y_max = 700
    return BoundingBox(x_min, x_max, y_min, y_max)

def y_affine_positive_(np_arr_fixture: NDArray, bounding_box_fixture: BoundingBox):
    resolution = 100
    return RasterNp(
        np_arr_fixture, bounding_box_fixture, resolution, y_affine_negative=False, proj4_code=PROJ4_CODE_FIXTURE)
