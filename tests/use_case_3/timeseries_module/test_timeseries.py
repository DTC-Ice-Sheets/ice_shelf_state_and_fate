"""Module for creating a timeseries object."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationStrategy,
    NormalisationType,
    AggregationPeriod,
    # PlotSpreadType,
    TimeseriesType,
    TimeseriesRegion,
    UncertaintyType,
)


@pytest.fixture
def timeseries_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    uncertainties = pd.Series(
        [10, 11, 12, 13, 14],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return timeseries, uncertainties, "time series 1", "unit 1", "measurement 1", "cumulative", "ice_shelf"


@pytest.fixture
def timeseries_obj_fixture():
    return Timeseries(
        label="test",
        measurement_name="measurement 1",
        timeseries_type=TimeseriesType.CUMULATIVE,
        timeseries_region=TimeseriesRegion("ice_shelf"),
    )


@pytest.fixture
def data_array_3d_fixture():
    nested_list = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    ]
    return np.array(nested_list)


@pytest.fixture
def data_array_2d_fixture():
    nested_list = [[1, 2], [3, 4]]
    return np.array(nested_list)


@pytest.fixture
def data_array_1d_fixture():
    nested_list = [
        [1, 2, 3, 4],
    ]
    return np.array(nested_list)


@pytest.fixture
def dimension_names_3d_fixture():
    return ["X", "Y", "time"]


@pytest.fixture
def dimension_names_ordered_3d_fixture():
    return ["time", "X", "Y"]


@pytest.fixture
def dimension_names_2d_fixture():
    return ["X", "Y"]


@pytest.fixture
def dimension_names_1d_fixture():
    return ["X"]


@pytest.fixture
def mask_fixture():
    nested_list = [[True, False, False], [False, False, False], [False, False, False]]
    return np.array(nested_list)


@pytest.fixture
def data_array_nans_fixture():
    nested_list = [
        [[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, np.nan, 12], [13, 14, 15], [16, np.nan, 18]],
        [[19, 20, np.nan], [np.nan, 23, 24], [np.nan, 26, 27]],
    ]
    return np.array(nested_list)


@pytest.fixture
def data_array_uncertainties_nans_fixture():
    nested_list = [
        [[np.nan, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        [[0.5, np.nan, 0.5], [0.5, 0.5, 0.5], [0.5, np.nan, 0.5]],
        [[0.5, 0.5, np.nan], [np.nan, 0.5, 0.5], [np.nan, 0.5, 0.5]],
    ]
    return np.array(nested_list)


@pytest.fixture
def dimension_data_fixture_2d():
    return {"X": np.array([0, 1, 2]), "Y": np.array([0, 1, 2])}


@pytest.fixture
def dimension_data_fixture():
    return {"X": np.array([0, 1, 2]), "Y": np.array([0, 1, 2]), "time": np.array([10, 11, 12])}


@pytest.fixture
def dimension_data_fixture_dates():
    return {
        "X": np.array([0, 1, 2]),
        "Y": np.array([0, 1, 2]),
        "time": np.array(["2012-01-01", "2012-03-01", "2014-01-01"]),
    }


@pytest.fixture
def time_index_fixture():
    return {0: [0], 1: [1], 2: [2]}


@pytest.fixture
def uncertainty_vector_fixture():
    return np.array([0.4, 0.5, 0.6])


@pytest.fixture
def time_series_fixture():
    return pd.DataFrame(
        np.array([[6, 0.5, 100], [7, 0.5, 75], [8, 0.5, 50]]), columns=["aggregation", "uncertainty", "coverage"]
    )


@pytest.fixture
def timeseries_dataset_fixture(timeseries_fixture):
    ds = xr.Dataset(
        data_vars={
            "measurement": (["time", "y", "x"], timeseries_fixture[0].values.reshape((-1, 1, 1)).astype(float)),
            "measurement_uncertainties": (
                ["time", "y", "x"],
                timeseries_fixture[1].values.reshape((-1, 1, 1)).astype(float),
            ),
        },
        coords={
            "time": timeseries_fixture[0].index.values,
            "x": [0],
            "y": [1],
        },
    )

    return ds, "time series 1", "unit 1", "measurement", "cumulative", "ice_shelf"


def test_process_timeseries_1d(timeseries_dataset_fixture):
    timeseries = Timeseries(
        label=timeseries_dataset_fixture[1],
        measurement_name=timeseries_dataset_fixture[2],
        unit=timeseries_dataset_fixture[2],
        timeseries_type=TimeseriesType(timeseries_dataset_fixture[4]),
        timeseries_region=TimeseriesRegion(timeseries_dataset_fixture[5]),
    )

    mask_geometries = {"ice_shelf": None, "ocean": None, "climate": None}

    timeseries.process_timeseries(
        data=timeseries_dataset_fixture[0],
        measurement_name=timeseries_dataset_fixture[3],
        measurement_uncertainty_name="measurement_uncertainties",
        mask_geometries=mask_geometries,
        apply_mask=False,
        period=AggregationPeriod.MONTH,
        aggregate=AggregationStrategy.MEAN,
        uncertainty=UncertaintyType.STD,
        independent=True,
        t0_normalisation=None,
    )

    assert timeseries.label == timeseries_dataset_fixture[1]
    assert np.allclose(timeseries.measurements.values, timeseries_dataset_fixture[0]["measurement"].values.ravel())
    assert np.allclose(timeseries.measurement_std.values, np.zeros(timeseries.measurements.values.shape))
    assert np.allclose(timeseries.measurement_n.values, np.ones(timeseries.measurements.values.shape))
    assert np.allclose(
        timeseries.measurement_uncertainties.values,
        timeseries_dataset_fixture[0]["measurement_uncertainties"].values.ravel(),
    )
    assert np.allclose(timeseries.measurements.values, timeseries_dataset_fixture[0]["measurement"].values.ravel())
    assert np.allclose(
        timeseries.measurement_uncertainties_std.values, np.zeros(timeseries.measurement_uncertainties.values.shape)
    )
    assert np.allclose(
        timeseries.measurement_uncertainties_n.values, np.ones(timeseries.measurement_uncertainties.values.shape)
    )


def test_process_timeseries_3d(timeseries_dataset_fixture):
    timeseries = Timeseries(
        label=timeseries_dataset_fixture[1],
        measurement_name=timeseries_dataset_fixture[2],
        unit=timeseries_dataset_fixture[2],
        timeseries_type=TimeseriesType(timeseries_dataset_fixture[4]),
        timeseries_region=TimeseriesRegion(timeseries_dataset_fixture[5]),
    )

    mask_geometries = {"ice_shelf": None, "ocean": None, "climate": None}

    timeseries.process_timeseries(
        data=timeseries_dataset_fixture[0],
        measurement_name=timeseries_dataset_fixture[3],
        measurement_uncertainty_name="measurement_uncertainties",
        mask_geometries=mask_geometries,
        apply_mask=False,
        period=AggregationPeriod.MONTH,
        aggregate=AggregationStrategy.MEAN,
        uncertainty=UncertaintyType.STD,
        independent=True,
        t0_normalisation=None,
    )

    assert timeseries.label == timeseries_dataset_fixture[1]
    assert np.allclose(timeseries.measurements.values, timeseries_dataset_fixture[0]["measurement"].values.ravel())
    assert np.allclose(timeseries.measurement_std.values, np.zeros(timeseries.measurements.values.shape))
    assert np.allclose(timeseries.measurement_n.values, np.ones(timeseries.measurements.values.shape))
    assert np.allclose(
        timeseries.measurement_uncertainties.values,
        timeseries_dataset_fixture[0]["measurement_uncertainties"].values.ravel(),
    )
    assert np.allclose(timeseries.measurements.values, timeseries_dataset_fixture[0]["measurement"].values.ravel())
    assert np.allclose(
        timeseries.measurement_uncertainties_std.values, np.zeros(timeseries.measurement_uncertainties.values.shape)
    )
    assert np.allclose(
        timeseries.measurement_uncertainties_n.values, np.ones(timeseries.measurement_uncertainties.values.shape)
    )


def test_timeseries_type():
    assert TimeseriesType.CUMULATIVE.value == "cumulative"
    assert TimeseriesType.DERIVATIVE.value == "derivative"


def test_timeseries(timeseries_fixture):
    timeseries = Timeseries(
        label=timeseries_fixture[2],
        measurement_name=timeseries_fixture[4],
        measurements=timeseries_fixture[0],
        measurement_uncertainties=timeseries_fixture[1],
        unit=timeseries_fixture[3],
        timeseries_type=TimeseriesType(timeseries_fixture[5]),
        timeseries_region=TimeseriesRegion(timeseries_fixture[6]),
    )

    assert timeseries.label == "time series 1"
    assert timeseries.unit == "unit 1"
    assert timeseries.measurement_name == "measurement 1"
    assert timeseries.measurements.equals(timeseries_fixture[0])
    assert timeseries.measurement_uncertainties.equals(timeseries_fixture[1])
    assert timeseries.timeseries_type == TimeseriesType("cumulative")
    assert timeseries.timeseries_region == TimeseriesRegion("ice_shelf")

    timeseries = Timeseries(
        label=timeseries_fixture[2],
        measurement_name=timeseries_fixture[4],
        measurements=timeseries_fixture[0],
        unit=timeseries_fixture[3],
        timeseries_type=TimeseriesType(timeseries_fixture[5]),
        timeseries_region=TimeseriesRegion("ocean"),
    )

    assert timeseries.label == "time series 1"
    assert timeseries.unit == "unit 1"
    assert timeseries.measurements.equals(timeseries_fixture[0])
    assert timeseries.measurement_uncertainties is None
    assert timeseries.timeseries_type == TimeseriesType("cumulative")
    assert timeseries.timeseries_region == TimeseriesRegion("ocean")


def test__move_axes_time(data_array_3d_fixture, dimension_names_3d_fixture):
    data_reordered = Timeseries._move_axes_time(data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture)

    assert data_reordered[0, 0, 0] == 1
    assert data_reordered[2, 0, 0] == 3
    assert data_reordered[0, 2, 1] == 22


def test__move_axes_time_labels(dimension_names_3d_fixture):
    dimensions_reordered = Timeseries._move_axes_time_labels(dimension_names=dimension_names_3d_fixture)
    assert dimensions_reordered == ["time", "X", "Y"]


def test__expand_dimensions1d(timeseries_obj_fixture, data_array_1d_fixture, dimension_names_1d_fixture):
    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_1d_fixture, dimension_names=dimension_names_1d_fixture, modify_dimension_names=False
    )

    assert data_modified.shape == (1, 1, 4)
    assert dimension_name_modified == ["X"]

    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_1d_fixture, dimension_names=dimension_names_1d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (1, 1, 4)
    assert dimension_name_modified == ["blank", "blank", "X"]


def test__expand_dimensions2d(timeseries_obj_fixture, data_array_2d_fixture, dimension_names_2d_fixture):
    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_2d_fixture, dimension_names=dimension_names_2d_fixture, modify_dimension_names=False
    )
    assert data_modified.shape == (1, 2, 2)
    assert dimension_name_modified == ["X", "Y"]

    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_2d_fixture, dimension_names=dimension_names_2d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (1, 2, 2)
    assert dimension_name_modified == ["blank", "X", "Y"]


def test__expand_dimensions3d(timeseries_obj_fixture, data_array_3d_fixture, dimension_names_3d_fixture):
    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=False
    )

    assert data_modified.shape == (3, 3, 3)
    assert dimension_name_modified == ["X", "Y", "time"]

    data_modified, dimension_name_modified = timeseries_obj_fixture._expand_dimensions(
        data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (3, 3, 3)
    assert dimension_name_modified == ["X", "Y", "time"]


def test__find_time_indexes(timeseries_obj_fixture, dimension_data_fixture_dates):
    time_idx = timeseries_obj_fixture._find_time_indexes(
        time_coord=pd.Series(dimension_data_fixture_dates["time"], name="time"), period=AggregationPeriod.YEAR
    )
    keys = [2012, 2014]

    assert list(time_idx.keys()) == keys
    assert time_idx[2012].equals(pd.Index([0, 1]))
    assert time_idx[2014].equals(pd.Index([2]))

    time_idx = timeseries_obj_fixture._find_time_indexes(
        time_coord=pd.Series(dimension_data_fixture_dates["time"], name="time"), period=AggregationPeriod.MONTH
    )
    keys = ["2012-1", "2012-3", "2014-1"]
    keys = [
        pd.Timestamp(year=2012, month=1, day=1),
        pd.Timestamp(year=2012, month=3, day=1),
        pd.Timestamp(year=2014, month=1, day=1),
    ]
    assert list(time_idx.keys()) == keys
    assert time_idx[pd.Timestamp(year=2012, month=1, day=1)].equals(pd.Index([0]))
    assert time_idx[pd.Timestamp(year=2012, month=3, day=1)].equals(pd.Index([1]))
    assert time_idx[pd.Timestamp(year=2014, month=1, day=1)].equals(pd.Index([2]))


def test__process_time_indexes_2d(timeseries_obj_fixture, dimension_names_2d_fixture, dimension_data_fixture_2d):
    time_indexes = timeseries_obj_fixture._process_time_indexes(
        dimension_names=dimension_names_2d_fixture, dimension_data=dimension_data_fixture_2d, period=None
    )

    assert time_indexes == {0: [0]}


def test__process_time_indexes(
    timeseries_obj_fixture, dimension_names_3d_fixture, dimension_data_fixture, dimension_data_fixture_dates
):
    time_idx = timeseries_obj_fixture._process_time_indexes(
        dimension_names=dimension_names_3d_fixture, dimension_data=dimension_data_fixture, period=None
    )
    assert time_idx == {10: [0], 11: [1], 12: [2]}

    time_idx = timeseries_obj_fixture._process_time_indexes(
        dimension_names=dimension_names_3d_fixture,
        dimension_data=dimension_data_fixture_dates,
        period=AggregationPeriod.YEAR,
    )

    keys = [2012, 2014]

    assert list(time_idx.keys()) == keys
    assert time_idx[2012].equals(pd.Index([0, 1]))
    assert time_idx[2014].equals(pd.Index([2]))


def test_count_nan_elements(data_array_3d_fixture, data_array_nans_fixture, time_index_fixture):
    # assert np.array_equal(
    #     Timeseries._count_nan_elements(data=data_array_3d_fixture, non_nan=False),
    #     np.array([0]),
    # )

    assert np.array_equal(
        Timeseries._count_nan_elements(data=data_array_3d_fixture, non_nan=False, time_index=time_index_fixture),
        np.array([0, 0, 0]),
    )

    assert np.array_equal(
        Timeseries._count_nan_elements(data=data_array_3d_fixture, non_nan=True, time_index=time_index_fixture),
        np.array([9, 9, 9]),
    )

    assert np.array_equal(
        Timeseries._count_nan_elements(data=data_array_nans_fixture, non_nan=False, time_index=time_index_fixture),
        np.array([1, 2, 3]),
    )

    assert np.array_equal(
        Timeseries._count_nan_elements(data=data_array_nans_fixture, non_nan=True, time_index=time_index_fixture),
        np.array([8, 7, 6]),
    )


def test_spread_calculation_type():
    assert Timeseries._spread_calculation_type(AggregationStrategy("mean")) == (AggregationStrategy.STD, True)
    assert Timeseries._spread_calculation_type(AggregationStrategy("median")) == (AggregationStrategy.IQR, True)
    assert Timeseries._spread_calculation_type(AggregationStrategy("max")) == (AggregationStrategy.NONE, False)
    assert Timeseries._spread_calculation_type(AggregationStrategy("min")) == (AggregationStrategy.NONE, False)


def test_mask_data_cube(
    timeseries_obj_fixture, data_array_nans_fixture, mask_fixture, dimension_names_ordered_3d_fixture
):
    data_masked, modify = timeseries_obj_fixture._mask_data_cube(
        data_cube=data_array_nans_fixture,
        data_mask=mask_fixture,
        dimension_names=dimension_names_ordered_3d_fixture,
        modify=True,
    )

    assert np.array_equal(
        np.array(
            [
                [[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[np.nan, np.nan, 12], [13, 14, 15], [16, np.nan, 18]],
                [[np.nan, 20, np.nan], [np.nan, 23, 24], [np.nan, 26, 27]],
            ]
        ),
        data_masked,
        equal_nan=True,
    )

    assert modify is False


def test__aggregate_statistic(
    timeseries_obj_fixture,
    data_array_nans_fixture,
    dimension_names_3d_fixture,
    time_index_fixture,
):
    data, dimension_names = timeseries_obj_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = timeseries_obj_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    dimensions_reordered = Timeseries._move_axes_time_labels(dimension_names=dimension_names)

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("mean"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([11.5, 14, 14.25]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("median"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([11.5, 14, 13.5]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("max"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([19, 26, 27]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("min"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([4, 2, 3]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("std"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([5.123, 8.635, 7.902]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_statistic(
            data=data_reordered,
            aggregate=AggregationStrategy("IQR"),
            time_index=time_index_fixture,
            dimension_names=dimensions_reordered,
        )[0],
        np.array([7.5, 15, 11.25]),
        rtol=1e-2,
    )


def test__aggregate_uncertainties_mean(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = timeseries_obj_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = timeseries_obj_fixture._move_axes_time(data=data, dimension_names=dimension_names)

    assert np.allclose(
        timeseries_obj_fixture._aggregate_uncertainties_mean(
            data_uncertainties=data_reordered,
            uncertainty=UncertaintyType.STD,
            time_index=time_index_fixture,
            independent=True,
        )[0],
        np.array([5.140, 6.217, 5.761]),
        rtol=1e-2,
    )

    assert np.allclose(
        timeseries_obj_fixture._aggregate_uncertainties_mean(
            data_uncertainties=data_reordered,
            uncertainty=UncertaintyType.STD,
            time_index=time_index_fixture,
            independent=False,
        )[0],
        np.array([11.5, 14, 14.25]),
        rtol=1e-2,
    )


def test__aggregate_uncertainties_max(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = timeseries_obj_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = timeseries_obj_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    uncertainty = timeseries_obj_fixture._aggregate_uncertainties_minmax(
        data_observations=data_reordered,
        data_uncertainties=data_reordered,
        uncertainty=UncertaintyType.MAX,
        time_index=time_index_fixture,
    )

    assert np.array_equal(uncertainty[0], np.array([19, 26, 27]))
    assert np.array_equal(uncertainty[1], np.array([0, 1, 2]))


def test__aggregate_uncertainties_min(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = timeseries_obj_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = timeseries_obj_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    uncertainty = timeseries_obj_fixture._aggregate_uncertainties_minmax(
        data_observations=data_reordered,
        data_uncertainties=data_reordered,
        uncertainty=UncertaintyType.MIN,
        time_index=time_index_fixture,
    )

    assert np.array_equal(uncertainty[0], np.array([4, 2, 3]))
    assert np.array_equal(uncertainty[1], np.array([0, 1, 2]))


def test__calculate_aggregate_timeseries_3d_mean(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, dimension_data_fixture
):
    data_df = timeseries_obj_fixture._calculate_aggregate_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_names_3d_fixture,
        aggregate=AggregationStrategy.MEAN,
    )
    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["aggregation", "data_std", "aggregation_n"]
    assert np.allclose(data_df["aggregation"].values, np.array([11.5, 14, 14.25]), rtol=1e-2)
    assert np.allclose(data_df["data_std"].values, np.array([5.123, 8.635, 7.902]), rtol=1e-2)
    assert np.allclose(data_df["aggregation_n"].values, np.array([6, 7, 8]), rtol=1e-2)


def test__calculate_aggregate_timeseries_3d_median(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, dimension_data_fixture
):
    data_df = timeseries_obj_fixture._calculate_aggregate_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_names_3d_fixture,
        aggregate=AggregationStrategy.MEDIAN,
    )

    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["aggregation", "data_IQR", "aggregation_n"]


def test__calculate_aggregate_timeseries_3d_max(
    timeseries_obj_fixture, data_array_nans_fixture, dimension_names_3d_fixture, dimension_data_fixture
):
    data_df = timeseries_obj_fixture._calculate_aggregate_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_names_3d_fixture,
        aggregate=AggregationStrategy.MAX,
    )
    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["aggregation", "data_None", "aggregation_n"]
    assert np.allclose(data_df["aggregation"].values, np.array([19, 26, 27]), rtol=1e-2)
    assert np.allclose(data_df["data_None"].values, np.array([0, 0, 0]), rtol=1e-2)
    assert np.allclose(data_df["aggregation_n"].values, np.array([0, 0, 0]), rtol=1e-2)


def test__combine_uncertainties_quadrature(uncertainty_vector_fixture):
    assert np.allclose(
        Timeseries._combine_uncertainties_quadrature(
            data_uncertainties_1=uncertainty_vector_fixture, data_uncertainties_2=np.array([0.4])
        ),
        np.array([0.566, 0.640, 0.721]),
        rtol=1e-2,
    )

    assert np.allclose(
        Timeseries._combine_uncertainties_quadrature(
            data_uncertainties_1=uncertainty_vector_fixture, data_uncertainties_2=uncertainty_vector_fixture
        ),
        np.array([0.566, 0.707, 0.849]),
        rtol=1e-2,
    )


def test__calculate_coverage_timeseries(
    timeseries_obj_fixture,
    data_array_nans_fixture,
    mask_fixture,
    dimension_data_fixture_dates,
    dimension_names_ordered_3d_fixture,
):
    coverage = timeseries_obj_fixture._calculate_coverage_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        data_mask=mask_fixture,
        period=AggregationPeriod.MONTH,
    )

    assert np.allclose(np.array([[100, 8], [87.5, 7], [75, 6]]), coverage)


def test__t0_normalisation(timeseries_obj_fixture, data_array_1d_fixture):
    data_norm = timeseries_obj_fixture._t0_normalisation(
        data_observation=data_array_1d_fixture[0], normalisation_type=NormalisationType("first")
    )
    assert np.array_equal(data_norm, np.array([0, 1, 2, 3]))

    data_norm = timeseries_obj_fixture._t0_normalisation(
        data_observation=data_array_1d_fixture[0], normalisation_type=NormalisationType("linear_regression")
    )
    assert np.allclose(data_norm, np.array([0, 1, 2, 3]))


def test_t0_normalisation_uncertainties(timeseries_obj_fixture, uncertainty_vector_fixture):
    data_norm = timeseries_obj_fixture._t0_normalisation_uncertainties(
        data_uncertainties=uncertainty_vector_fixture, normalisation_type=NormalisationType("first")
    )
    assert np.allclose(
        data_norm,
        np.array([0.566, 0.640, 0.721]),
        rtol=1e-2,
    )


def test__propagate_uncertainties_timeseries(
    timeseries_obj_fixture,
    data_array_nans_fixture,
    data_array_uncertainties_nans_fixture,
    dimension_names_ordered_3d_fixture,
    dimension_data_fixture_dates,
):
    result = timeseries_obj_fixture._propagate_uncertainties_timeseries(
        data_observations=data_array_nans_fixture,
        data_uncertainties=data_array_uncertainties_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        aggregate=AggregationStrategy("mean"),
        uncertainty=UncertaintyType("std"),
        period=AggregationPeriod.YEAR,
        independent=True,
        t0_normalisation=None,
    )

    assert np.allclose(
        result["uncertainty"].values,
        np.array([0.129, 0.204]),
        rtol=1e-2,
    )
    assert np.allclose(
        result["uncertainty_std"].values,
        np.array([0, 0]),
        rtol=1e-2,
    )
    assert np.allclose(result["uncertainty_n"].values, np.array([15, 6]))
    assert result.index.to_list() == [2012, 2014]


def test__determine_coverage(timeseries_obj_fixture, mask_fixture, data_array_nans_fixture, time_index_fixture):
    coverage, time_coords_agg = timeseries_obj_fixture._determine_coverage(
        data_observations=data_array_nans_fixture, mask_raster=mask_fixture, time_index=time_index_fixture
    )

    assert np.allclose(coverage, np.array([1, 0.875, 0.750]))
    assert time_coords_agg == [0, 1, 2]


def test__determine_coverage_point(timeseries_obj_fixture, mask_fixture, data_array_nans_fixture):
    assert 0.875 == pytest.approx(
        timeseries_obj_fixture._determine_coverage_point(
            data_observations=data_array_nans_fixture, mask_raster=mask_fixture
        )
    )


def test_determine_count_pixels(mask_fixture, data_array_nans_fixture):
    assert Timeseries._count_pixels(mask_fixture) == 8
    assert Timeseries._count_pixels(data_array_nans_fixture) == 21


def test_summary_statistics(timeseries_obj_fixture, time_series_fixture):
    timeseries_obj_fixture.measurements = time_series_fixture["aggregation"]
    timeseries_obj_fixture.measurement_uncertainties = time_series_fixture["uncertainty"]
    timeseries_obj_fixture.measurement_coverage = time_series_fixture["coverage"]
    timeseries_obj_fixture.unit = "m"

    timeseries_obj_fixture.summary_statistics()

    summary_stats = timeseries_obj_fixture.summary_stats

    assert summary_stats.loc["test - Initial Value (m)", "Value"] == 6
    assert summary_stats.loc["test - Initial Value (m)", "Lower Uncertainty Bound"] == 5.5
    assert summary_stats.loc["test - Initial Value (m)", "Upper Uncertainty Bound"] == 6.5

    assert summary_stats.loc["test - Final Value (m)", "Value"] == 8
    assert summary_stats.loc["test - Final Value (m)", "Lower Uncertainty Bound"] == 7.5
    assert summary_stats.loc["test - Final Value (m)", "Upper Uncertainty Bound"] == 8.5

    assert summary_stats.loc["test - Difference (m)", "Value"] == 2
    assert np.isclose(summary_stats.loc["test - Difference (m)", "Lower Uncertainty Bound"], 2 - 0.707107)
    assert np.isclose(summary_stats.loc["test - Difference (m)", "Upper Uncertainty Bound"], 2 + 0.707107)

    assert summary_stats.loc["test - Maximum Value (m)", "Value"] == 8
    assert summary_stats.loc["test - Maximum Value (m)", "Lower Uncertainty Bound"] == 7.5
    assert summary_stats.loc["test - Maximum Value (m)", "Upper Uncertainty Bound"] == 8.5

    assert summary_stats.loc["test - Minimum Value (m)", "Value"] == 6
    assert summary_stats.loc["test - Minimum Value (m)", "Lower Uncertainty Bound"] == 5.5
    assert summary_stats.loc["test - Minimum Value (m)", "Upper Uncertainty Bound"] == 6.5

    assert summary_stats.loc["test - 10th Percentile Value (m)", "Value"] == 6.2
    assert summary_stats.loc["test - 25th Percentile Value (m)", "Value"] == 6.5
    assert summary_stats.loc["test - 50th Percentile Value (m)", "Value"] == 7
    assert summary_stats.loc["test - 75th Percentile Value (m)", "Value"] == 7.5
    assert summary_stats.loc["test - 90th Percentile Value (m)", "Value"] == 7.8

    assert summary_stats.loc["Coverage Maximum (%)", "Value"] == 100.0
    assert summary_stats.loc["Coverage Minimum (%)", "Value"] == 50.0
    assert summary_stats.loc["Coverage Range (%)", "Value"] == 50.0
    assert summary_stats.loc["Coverage Mean (%)", "Value"] == 75.0
    assert summary_stats.loc["Coverage Median (%)", "Value"] == 75.0
    assert np.isnan(summary_stats.loc["Observation/Uncertainty Correlation", "Value"])

    assert np.isclose(summary_stats.loc["Observation Linearity (correlation coef.)", "Value"], 1.0)


def test__calculate_percentiles_timeseries(
    timeseries_obj_fixture,
    data_array_nans_fixture,
    dimension_data_fixture_dates,
    dimension_names_ordered_3d_fixture,
):
    percentiles = timeseries_obj_fixture._calculate_percentiles_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        period=AggregationPeriod.MONTH,
        percentiles=(25, 75),
    )

    assert np.allclose(np.array([[3.75, 7.25, 8], [12.5, 15.5, 7], [20.75, 25.5, 6]]), percentiles.values)


def test__determine_percentiles(timeseries_obj_fixture, data_array_nans_fixture, time_index_fixture):
    percentiles, time_coords_agg = timeseries_obj_fixture._determine_percentiles(
        data_observations=data_array_nans_fixture, time_index=time_index_fixture, percentile=25
    )

    assert np.allclose(percentiles, np.array([3.75, 12.5, 20.75]))
    assert time_coords_agg == [0, 1, 2]
