"""Module for testing TimeSeriesModule."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.timeseries_module.timeseries_module import TimeSeriesModule


@pytest.fixture
def time_series_module_fixture():
    return TimeSeriesModule()


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


def test__move_axes_time(data_array_3d_fixture, dimension_names_3d_fixture):
    data_reordered = TimeSeriesModule._move_axes_time(
        data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture
    )

    assert data_reordered[0, 0, 0] == 1
    assert data_reordered[2, 0, 0] == 3
    assert data_reordered[0, 2, 1] == 22


def test__move_axes_time_labels(dimension_names_3d_fixture):
    dimensions_reordered = TimeSeriesModule._move_axes_time_labels(dimension_names=dimension_names_3d_fixture)
    assert dimensions_reordered == ["time", "X", "Y"]


def test__expand_dimensions1d(time_series_module_fixture, data_array_1d_fixture, dimension_names_1d_fixture):
    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_1d_fixture, dimension_names=dimension_names_1d_fixture, modify_dimension_names=False
    )

    assert data_modified.shape == (1, 1, 4)
    assert dimension_name_modified == ["X"]

    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_1d_fixture, dimension_names=dimension_names_1d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (1, 1, 4)
    assert dimension_name_modified == ["blank", "blank", "X"]


def test__expand_dimensions2d(time_series_module_fixture, data_array_2d_fixture, dimension_names_2d_fixture):
    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_2d_fixture, dimension_names=dimension_names_2d_fixture, modify_dimension_names=False
    )
    assert data_modified.shape == (1, 2, 2)
    assert dimension_name_modified == ["X", "Y"]

    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_2d_fixture, dimension_names=dimension_names_2d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (1, 2, 2)
    assert dimension_name_modified == ["blank", "X", "Y"]


def test__expand_dimensions3d(time_series_module_fixture, data_array_3d_fixture, dimension_names_3d_fixture):
    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=False
    )

    assert data_modified.shape == (3, 3, 3)
    assert dimension_name_modified == ["X", "Y", "time"]

    data_modified, dimension_name_modified = time_series_module_fixture._expand_dimensions(
        data=data_array_3d_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )

    assert data_modified.shape == (3, 3, 3)
    assert dimension_name_modified == ["X", "Y", "time"]


def test_find_time_indexes(time_series_module_fixture, dimension_data_fixture_dates):
    time_idx = time_series_module_fixture.find_time_indexes(
        time_coord=pd.Series(dimension_data_fixture_dates["time"], name="time"), period="year"
    )
    keys = [2012, 2014]

    assert list(time_idx.keys()) == keys
    assert time_idx[2012].equals(pd.Index([0, 1]))
    assert time_idx[2014].equals(pd.Index([2]))

    time_idx = time_series_module_fixture.find_time_indexes(
        time_coord=pd.Series(dimension_data_fixture_dates["time"], name="time"), period="month"
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


def test_process_time_indexes_2d(time_series_module_fixture, dimension_names_2d_fixture, dimension_data_fixture_2d):
    time_indexes = time_series_module_fixture.process_time_indexes(
        dimension_names=dimension_names_2d_fixture, dimension_data=dimension_data_fixture_2d, period=None
    )

    assert time_indexes == {0: [0]}


def test_process_time_indexes(
    time_series_module_fixture, dimension_names_3d_fixture, dimension_data_fixture, dimension_data_fixture_dates
):
    time_idx = time_series_module_fixture.process_time_indexes(
        dimension_names=dimension_names_3d_fixture, dimension_data=dimension_data_fixture, period=None
    )
    assert time_idx == {10: [0], 11: [1], 12: [2]}

    time_idx = time_series_module_fixture.process_time_indexes(
        dimension_names=dimension_names_3d_fixture, dimension_data=dimension_data_fixture_dates, period="year"
    )

    keys = [2012, 2014]

    assert list(time_idx.keys()) == keys
    assert time_idx[2012].equals(pd.Index([0, 1]))
    assert time_idx[2014].equals(pd.Index([2]))


def test_count_nan_elements(data_array_3d_fixture, data_array_nans_fixture, time_index_fixture):
    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_3d_fixture, non_nan=False),
        np.array([0]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_3d_fixture, non_nan=False, time_index=time_index_fixture),
        np.array([0, 0, 0]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_3d_fixture, non_nan=True, time_index=time_index_fixture),
        np.array([9, 9, 9]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_nans_fixture, non_nan=False, time_index=time_index_fixture),
        np.array([1, 2, 3]),
    )

    assert np.array_equal(
        TimeSeriesModule.count_nan_elements(data=data_array_nans_fixture, non_nan=True, time_index=time_index_fixture),
        np.array([8, 7, 6]),
    )


def test_spread_calculation_type():
    assert TimeSeriesModule.spread_calculation_type("mean") == ("std", True)
    assert TimeSeriesModule.spread_calculation_type("median") == ("IQR", True)
    assert TimeSeriesModule.spread_calculation_type("max") == (None, False)
    assert TimeSeriesModule.spread_calculation_type("min") == (None, False)


def test_mask_data_cube(
    time_series_module_fixture, data_array_nans_fixture, mask_fixture, dimension_names_ordered_3d_fixture
):
    # print(data_array_nans_fixture.shape, np.squeeze(mask_fixture).shape)
    data_masked, modify = time_series_module_fixture.mask_data_cube(
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


def test_aggregate_statistic(
    time_series_module_fixture,
    data_array_nans_fixture,
    dimension_names_3d_fixture,
    time_index_fixture,
):
    data, dimension_names = time_series_module_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = time_series_module_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    dimensions_reordered = TimeSeriesModule._move_axes_time_labels(dimension_names=dimension_names)

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="mean", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([11.5, 14, 14.25]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="median", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([11.5, 14, 13.5]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="max", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([19, 26, 27]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="min", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([4, 2, 3]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="std", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([5.123, 8.635, 7.902]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_statistic(
            data=data_reordered, aggregate="IQR", time_index=time_index_fixture, dimension_names=dimensions_reordered
        )[0],
        np.array([7.5, 15, 11.25]),
        rtol=1e-2,
    )


def test_aggregate_uncertainties_mean(
    time_series_module_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = time_series_module_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = time_series_module_fixture._move_axes_time(data=data, dimension_names=dimension_names)

    assert np.allclose(
        time_series_module_fixture.aggregate_uncertainties_mean(
            data_uncertainties=data_reordered, uncertainty="std", time_index=time_index_fixture, independent=True
        )[0],
        np.array([5.140, 6.217, 5.761]),
        rtol=1e-2,
    )

    assert np.allclose(
        time_series_module_fixture.aggregate_uncertainties_mean(
            data_uncertainties=data_reordered, uncertainty="std", time_index=time_index_fixture, independent=False
        )[0],
        np.array([11.5, 14, 14.25]),
        rtol=1e-2,
    )


def test_aggregate_uncertainties_max(
    time_series_module_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = time_series_module_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = time_series_module_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    uncertainty = time_series_module_fixture.aggregate_uncertainties_minmax(
        data_observations=data_reordered,
        data_uncertainties=data_reordered,
        uncertainty="max",
        time_index=time_index_fixture,
    )

    assert np.array_equal(uncertainty[0], np.array([19, 26, 27]))
    assert np.array_equal(uncertainty[1], np.array([0, 1, 2]))


def test_aggregate_uncertainties_min(
    time_series_module_fixture, data_array_nans_fixture, dimension_names_3d_fixture, time_index_fixture
):
    data, dimension_names = time_series_module_fixture._expand_dimensions(
        data=data_array_nans_fixture, dimension_names=dimension_names_3d_fixture, modify_dimension_names=True
    )
    data_reordered = time_series_module_fixture._move_axes_time(data=data, dimension_names=dimension_names)
    uncertainty = time_series_module_fixture.aggregate_uncertainties_minmax(
        data_observations=data_reordered,
        data_uncertainties=data_reordered,
        uncertainty="min",
        time_index=time_index_fixture,
    )

    assert np.array_equal(uncertainty[0], np.array([4, 2, 3]))
    assert np.array_equal(uncertainty[1], np.array([0, 1, 2]))


def test_calculate_aggregate_timeseries_3d_mean(
    time_series_module_fixture, data_array_nans_fixture, dimension_names_3d_fixture, dimension_data_fixture
):
    data_df = time_series_module_fixture.calculate_aggregate_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_names_3d_fixture,
        aggregate="mean",
    )
    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["aggregation", "data_std", "aggregation_n"]
    assert np.allclose(data_df["aggregation"].values, np.array([11.5, 14, 14.25]), rtol=1e-2)
    assert np.allclose(data_df["data_std"].values, np.array([5.123, 8.635, 7.902]), rtol=1e-2)
    assert np.allclose(data_df["aggregation_n"].values, np.array([6, 7, 8]), rtol=1e-2)


def test_calculate_aggregate_timeseries_3d_median(
    time_series_module_fixture, data_array_nans_fixture, dimension_names_3d_fixture, dimension_data_fixture
):
    data_df = time_series_module_fixture.calculate_aggregate_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture,
        dimension_names=dimension_names_3d_fixture,
        aggregate="median",
    )

    assert data_df.index.to_list() == [10, 11, 12]
    assert list(data_df.columns) == ["aggregation", "data_IQR", "aggregation_n"]


def test_combine_uncertainties_quadrature(uncertainty_vector_fixture):
    assert np.allclose(
        TimeSeriesModule.combine_uncertainties_quadrature(
            data_uncertainties_1=uncertainty_vector_fixture, data_uncertainties_2=np.array([0.4])
        ),
        np.array([0.566, 0.640, 0.721]),
        rtol=1e-2,
    )

    assert np.allclose(
        TimeSeriesModule.combine_uncertainties_quadrature(
            data_uncertainties_1=uncertainty_vector_fixture, data_uncertainties_2=uncertainty_vector_fixture
        ),
        np.array([0.566, 0.707, 0.849]),
        rtol=1e-2,
    )


def test_calculate_coverage_timeseries(
    time_series_module_fixture,
    data_array_nans_fixture,
    mask_fixture,
    dimension_data_fixture_dates,
    dimension_names_ordered_3d_fixture,
):
    coverage = time_series_module_fixture.calculate_coverage_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        data_mask=mask_fixture,
        period="month",
    )

    assert np.allclose(np.array([[100, 8], [87.5, 7], [75, 6]]), coverage)


def test_t0_normalisation(time_series_module_fixture, data_array_1d_fixture):
    data_norm = time_series_module_fixture.t0_normalisation(
        data_observation=data_array_1d_fixture[0], normalisation_type="first"
    )
    assert np.array_equal(data_norm, np.array([0, 1, 2, 3]))

    data_norm = time_series_module_fixture.t0_normalisation(
        data_observation=data_array_1d_fixture[0], normalisation_type="linear_regression"
    )
    assert np.allclose(data_norm, np.array([0, 1, 2, 3]))


def test_t0_normalisation_uncertainties(time_series_module_fixture, uncertainty_vector_fixture):
    data_norm = time_series_module_fixture.t0_normalisation_uncertainties(
        data_uncertainties=uncertainty_vector_fixture, normalisation_type="first"
    )
    assert np.allclose(
        data_norm,
        np.array([0.566, 0.640, 0.721]),
        rtol=1e-2,
    )


def test_propagate_uncertainties_timeseries(
    time_series_module_fixture,
    data_array_nans_fixture,
    data_array_uncertainties_nans_fixture,
    dimension_names_ordered_3d_fixture,
    dimension_data_fixture_dates,
):
    result = time_series_module_fixture.propagate_uncertainties_timeseries(
        data_observations=data_array_nans_fixture,
        data_uncertainties=data_array_uncertainties_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        aggregate="mean",
        uncertainty="std",
        period="year",
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


def test_determine_coverage(time_series_module_fixture, mask_fixture, data_array_nans_fixture, time_index_fixture):
    coverage, time_coords_agg = time_series_module_fixture.determine_coverage(
        data_observations=data_array_nans_fixture, mask_raster=mask_fixture, time_index=time_index_fixture
    )

    assert np.allclose(coverage, np.array([1, 0.875, 0.750]))
    assert time_coords_agg == [0, 1, 2]


def test_determine_coverage_point(time_series_module_fixture, mask_fixture, data_array_nans_fixture):
    assert 0.875 == pytest.approx(
        time_series_module_fixture.determine_coverage_point(
            data_observations=data_array_nans_fixture, mask_raster=mask_fixture
        )
    )


def test_determine_count_pixels(mask_fixture, data_array_nans_fixture):
    assert TimeSeriesModule.count_pixels(mask_fixture) == 8
    assert TimeSeriesModule.count_pixels(data_array_nans_fixture) == 21


def test_flatten_data_cube(
    time_series_module_fixture, data_array_3d_fixture, dimension_data_fixture, dimension_names_ordered_3d_fixture
):
    data_slices = time_series_module_fixture.flatten_data_cube(
        data_array_3d_fixture, dimension_data_fixture, dimension_names_ordered_3d_fixture
    )

    assert data_slices[0][0, 0] == 0
    assert data_slices[0][0, 1] == 0
    assert data_slices[0][0, 2] == 1

    assert data_slices[0][5, 0] == 1
    assert data_slices[0][5, 1] == 2
    assert data_slices[0][5, 2] == 6

    assert data_slices[2][0, 0] == 0
    assert data_slices[2][0, 1] == 0
    assert data_slices[2][0, 2] == 19

    assert data_slices[2][5, 0] == 1
    assert data_slices[2][5, 1] == 2
    assert data_slices[2][5, 2] == 24


def test_summary_statistics(time_series_module_fixture, time_series_fixture):
    summary_stats = time_series_module_fixture.summary_statistics(
        timeseries=time_series_fixture, observation_name="test", observation_units="m"
    )

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


def test_calculate_percentiles_timeseries(
    time_series_module_fixture,
    data_array_nans_fixture,
    dimension_data_fixture_dates,
    dimension_names_ordered_3d_fixture,
):
    percentiles = time_series_module_fixture.calculate_percentiles_timeseries(
        data_observations=data_array_nans_fixture,
        dimension_data=dimension_data_fixture_dates,
        dimension_names=dimension_names_ordered_3d_fixture,
        period="month",
        percentiles=(25, 75),
    )

    assert np.allclose(np.array([[3.75, 7.25, 8], [12.5, 15.5, 7], [20.75, 25.5, 6]]), percentiles.values)


def test_determine_percentiles():
    pass
