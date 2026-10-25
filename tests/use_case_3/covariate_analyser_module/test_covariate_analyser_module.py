"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.covariate_analyser_module import (
    CovariateAnalyser,
    JointPlotAnalysisType,
    ModelEvaluationMetric,
)
from use_case_3.covariate_analyser_module.timeseries_preprocessor import (
    TimeSeriesPreprocessor,
)
from use_case_3.timeseries_module.timeseries import Timeseries
from use_case_3.timeseries_module.timeseries_enums import (
    AggregationPeriod,
    AggregationStrategy,
    TimeseriesType,
)


@pytest.fixture
def timeseries_1_fixture():
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

    return Timeseries(
        measurements=timeseries,
        label="time series 1",
        unit="unit 1",
        measurement_name="measurement 1",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def expected_timeseries_dict_fixture1(timeseries_1_fixture):
    return {
        timeseries_1_fixture.label: timeseries_1_fixture,
    }


@pytest.fixture
def timeseries_2_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
            pd.Timestamp(year=2020, month=6, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 2",
        unit="unit 2",
        measurement_name="measurement 2",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def expected_timeseries_dict_fixture1_fixture_2(timeseries_1_fixture, timeseries_2_fixture):
    return {
        timeseries_1_fixture.label: timeseries_1_fixture,
        timeseries_2_fixture.label: timeseries_2_fixture,
    }


@pytest.fixture
def timeseries_3_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
            pd.Timestamp(year=2020, month=6, day=1),
            pd.Timestamp(year=2020, month=7, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 3",
        unit="unit 3",
        measurement_name="measurement 3",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_4_fixture():
    timeseries = pd.Series(
        [1, 2, 3, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=7, day=1),
            pd.Timestamp(year=2020, month=8, day=1),
            pd.Timestamp(year=2020, month=9, day=1),
            pd.Timestamp(year=2020, month=10, day=1),
            pd.Timestamp(year=2020, month=11, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 4",
        unit="unit 4",
        measurement_name="measurement 4",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_5_fixture():
    timeseries = pd.Series(
        [1, 2, 4, 5],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 5",
        unit="unit 5",
        measurement_name="measurement 5",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_1_inv_fixture():
    timeseries = pd.Series(
        [5, 4, 3, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 1 inv",
        unit="unit 1 inv",
        measurement_name="measurement 1 inv",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_6_fixture():
    timeseries = pd.Series(
        [-3, 1, 4, 5, 6],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 6",
        unit="unit 6",
        measurement_name="measurement 6",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_7_fixture():
    timeseries = pd.Series(
        [0, 1, 2, 1, 0],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )

    return Timeseries(
        measurements=timeseries,
        label="time series 7",
        unit="unit 7",
        measurement_name="measurement 7",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def timeseries_7_lag_fixture():
    timeseries = pd.Series(
        [1, 0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
            pd.Timestamp(year=2020, month=5, day=1),
        ],
    )
    return Timeseries(
        measurements=timeseries,
        label="time series 7 inv",
        unit="unit 7 inv",
        measurement_name="measurement 7",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def fixture_series_x():
    return Timeseries(
        measurements=pd.Series(
            [-2, -1, 0, 1.01, 2, 1, 0, -1, -2, -1, 0],
            name="x",
            index=pd.date_range(start="2020-01-01", periods=11, freq="MS"),
        ),
        measurement_name="x",
        label="x",
        unit="",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def fixture_series_y():
    return Timeseries(
        measurements=pd.Series(
            [-1, -2, -1, 0, 1, 2, 1, 0, -1, -2, -1],
            name="y",
            index=pd.date_range(start="2020-01-01", periods=11, freq="MS"),
        ),
        measurement_name="y",
        label="y",
        unit="",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def fixture_series_y2():
    return Timeseries(
        measurements=pd.Series(
            [-1, -2, -1, 0, 1, 2, 1, 0, -1, -1, -1],
            name="y",
            index=pd.date_range(start="2020-01-01", periods=11, freq="MS"),
        ),
        measurement_name="y",
        label="y",
        unit="",
        timeseries_type=TimeseriesType("cumulative"),
    )


@pytest.fixture
def fixture_data(fixture_series_x, fixture_series_y):
    return pd.concat([fixture_series_x.measurements, fixture_series_y.measurements], axis=1)


def test_correlation_coefficient(timeseries_1_fixture, timeseries_2_fixture, timeseries_7_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    covariate_analyser._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    covariate_analyser._timeseries_altered[timeseries_2_fixture.label] = timeseries_7_fixture
    corr, p_value = covariate_analyser.correlation_coefficient(
        timeseries_label_1=timeseries_1_fixture.label, timeseries_label_2=timeseries_2_fixture.label, altered=False
    )

    assert np.isclose(corr, 1)
    assert np.isclose(p_value, 0)

    corr, p_value = covariate_analyser.correlation_coefficient(
        timeseries_label_1=timeseries_1_fixture.label, timeseries_label_2=timeseries_2_fixture.label, altered=True
    )

    with pytest.raises(ValueError, match="The labels specified are not in the altered timeseries container"):
        corr, p_value = covariate_analyser.correlation_coefficient(
            timeseries_label_1=timeseries_1_fixture.label, timeseries_label_2=timeseries_7_fixture.label, altered=True
        )

    assert np.isclose(corr, 0)
    assert np.isclose(p_value, 1)


def test_find_correlated_vars(timeseries_1_fixture, timeseries_2_fixture, timeseries_7_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_7_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )
    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    correlated_vars = covariate_analyser.find_correlated_vars(corr_thresh=0.5)

    assert list(correlated_vars.keys()) == [(timeseries_1_fixture.label, timeseries_2_fixture.label)]
    assert next(iter(correlated_vars.values())) == (1.0, 0.0)


def test_find_crosscorrelated_vars(timeseries_7_fixture, timeseries_7_lag_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_7_fixture, timeseries_7_lag_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )
    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    correlated_vars = covariate_analyser.find_crosscorrelated_vars(corr_thresh=0.5, max_lag=1)

    assert list(correlated_vars.keys()) == [(timeseries_7_fixture.label, timeseries_7_lag_fixture.label)]
    assert np.isclose(next(iter(correlated_vars.values()))[0], 1.0)
    assert np.isclose(next(iter(correlated_vars.values()))[1], 0)
    assert np.isclose(next(iter(correlated_vars.values()))[2], -1.0)


def test_find_causal_vars(fixture_series_x, fixture_series_y2):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            fixture_series_x,
            fixture_series_y2,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=True,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    causal_vars = covariate_analyser.find_causal_vars(max_lags=3)

    assert next(iter(causal_vars.keys())) == ("y", "x")


def test_covariance(timeseries_1_fixture, timeseries_1_inv_fixture, timeseries_2_fixture, timeseries_7_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_1_inv_fixture, timeseries_7_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    covariate_analyser._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    covariate_analyser._timeseries_altered[timeseries_7_fixture.label] = timeseries_7_fixture

    corr = covariate_analyser.covariance(
        [timeseries_1_fixture.label, timeseries_1_fixture.label], normalisation="corrcoef", altered=False
    )
    assert np.allclose(corr, np.array([1]))

    corr = covariate_analyser.covariance(
        [timeseries_1_fixture.label, timeseries_2_fixture.label], normalisation="corrcoef", altered=False
    )
    assert np.allclose(corr, np.array([[np.nan, np.nan], [np.nan, np.nan]]), equal_nan=True)

    corr = covariate_analyser.covariance(
        [timeseries_1_fixture.label, timeseries_1_inv_fixture.label], normalisation="corrcoef", altered=False
    )
    assert np.allclose(corr, np.array([[1, -1], [-1, 1]]))

    corr = covariate_analyser.covariance(
        [timeseries_1_fixture.label, timeseries_7_fixture.label], normalisation="corrcoef", altered=True
    )
    assert np.allclose(corr, np.array([[1, 0], [0, 1]]))

    corr = covariate_analyser.covariance(
        [timeseries_1_fixture.label, timeseries_1_inv_fixture.label], normalisation="mean_centered", altered=False
    )
    assert np.allclose(corr, np.array([[2.5, -2.5], [-2.5, 2.5]]))


def test_max_crosscorrelation(
    timeseries_1_fixture, timeseries_1_inv_fixture, timeseries_7_fixture, timeseries_7_lag_fixture
):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            timeseries_1_fixture,
            timeseries_1_inv_fixture,
            timeseries_7_fixture,
            timeseries_7_lag_fixture,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    covariate_analyser._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    covariate_analyser._timeseries_altered[timeseries_1_inv_fixture.label] = timeseries_1_inv_fixture

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_1_fixture.label, timeseries_1_fixture.label, max_lag=None, altered=False
    )
    assert np.isclose(corr, 1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_1_fixture.label, timeseries_1_inv_fixture.label, max_lag=None, altered=False
    )
    assert np.isclose(corr, -1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_1_fixture.label, timeseries_1_fixture.label, max_lag=None, altered=True
    )
    assert np.isclose(corr, 1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_1_fixture.label, timeseries_1_inv_fixture.label, max_lag=None, altered=True
    )
    assert np.isclose(corr, -1)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_7_fixture.label, timeseries_7_lag_fixture.label, max_lag=0, altered=False
    )

    assert np.isclose(corr, 0)
    assert lag == 0

    corr, lag = covariate_analyser.max_crosscorrelation(
        timeseries_7_fixture.label, timeseries_7_lag_fixture.label, max_lag=1, altered=False
    )

    assert np.isclose(corr, 0.84515425)
    assert lag == -1


def test_align_timeseries(timeseries_7_fixture, timeseries_7_lag_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            timeseries_7_fixture,
            timeseries_7_lag_fixture,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    corr_lag, corr_lag_units = covariate_analyser._align_timeseries(
        timeseries_label_1=timeseries_7_fixture.label, timeseries_label_2=timeseries_7_lag_fixture.label, max_lag=1
    )

    timeseries_1_gt = pd.Series(
        [0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    timeseries_2_gt = pd.Series(
        [0, 1, 2, 1],
        index=[
            pd.Timestamp(year=2020, month=1, day=1),
            pd.Timestamp(year=2020, month=2, day=1),
            pd.Timestamp(year=2020, month=3, day=1),
            pd.Timestamp(year=2020, month=4, day=1),
        ],
    )

    assert np.allclose(
        covariate_analyser._timeseries_altered[timeseries_7_fixture.label].measurements.values, timeseries_1_gt.values
    )
    assert np.allclose(
        covariate_analyser._timeseries_altered[timeseries_7_lag_fixture.label].measurements.values,
        timeseries_2_gt.values,
    )
    assert corr_lag == -1
    assert corr_lag_units == AggregationPeriod.MONTH


def test_temporal_offset(timeseries_1_fixture, timeseries_6_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            timeseries_1_fixture,
            timeseries_6_fixture,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("mean")
        )[0]
        == 0.4
    )
    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("median")
        )[0]
        == -1
    )
    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("max")
        )[0]
        == 4
    )
    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("min")
        )[0]
        == -1
    )

    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("mean")
        )[1]
        is None
    )
    assert (
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("median")
        )[1]
        is None
    )
    assert np.array_equal(
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("max")
        )[1],
        0,
    )
    assert np.array_equal(
        covariate_analyser.temporal_offset(
            timeseries_1_fixture.label, timeseries_6_fixture.label, AggregationStrategy("min")
        )[1],
        2,
    )


def test_temporal_slope(timeseries_1_fixture, timeseries_1_inv_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            timeseries_1_fixture,
            timeseries_1_inv_fixture,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    model_fits = covariate_analyser.temporal_slope(timeseries_1_fixture.label, timeseries_1_inv_fixture.label)
    assert model_fits[0] == -1
    assert model_fits[1] == 1
    assert model_fits[2] == 1
    assert model_fits[3] == -1
    assert model_fits[4] == 5


def test_modelevaluation_metric__init():
    assert ModelEvaluationMetric("aic") == ModelEvaluationMetric.AIC
    assert ModelEvaluationMetric("bic") == ModelEvaluationMetric.BIC


def test_jointplotanalysistype__init():
    assert JointPlotAnalysisType("corr") == JointPlotAnalysisType.CORR
    assert JointPlotAnalysisType("cross_corr") == JointPlotAnalysisType.CROSS_CORR
    assert JointPlotAnalysisType("causal") == JointPlotAnalysisType.CAUSAL


@pytest.mark.parametrize(
    "analysis_type",
    [
        JointPlotAnalysisType.CORR,
        JointPlotAnalysisType.CROSS_CORR,
        JointPlotAnalysisType.CAUSAL,
    ],
)
def test_covariate_analysis_run_analysis(fixture_series_x, fixture_series_y, analysis_type):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            fixture_series_x,
            fixture_series_y,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=True,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    result = covariate_analyser.run_analysis(analysis_type=analysis_type)
    assert isinstance(result, dict)


def test_granger_causality_test_bestlag(fixture_data):
    best_lag, best_bic = CovariateAnalyser._granger_causality_test_bestlag(
        data=fixture_data, max_lags=3, metric=ModelEvaluationMetric("bic")
    )

    assert best_lag == 3
    np.testing.assert_almost_equal(best_bic, -64.1297785796557)


def test_granger_causality_test(fixture_series_x, fixture_series_y):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[
            fixture_series_x,
            fixture_series_y,
        ],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

    f_score, p_value = covariate_analyser._granger_causality_test(
        timeseries_label_1=fixture_series_x.label,
        timeseries_label_2=fixture_series_y.label,
        max_lags=3,
        metric=ModelEvaluationMetric.BIC,
    )

    np.testing.assert_almost_equal(f_score, 2.6001062494609055)
    np.testing.assert_almost_equal(p_value, 0.18902718453358755)


def test_power_model():
    x = np.array([1.0, 2.0, 3.0])

    result = CovariateAnalyser._power_model(x=x, a=2.0, b=2.0)

    np.testing.assert_allclose(result, np.array([2.0, 8.0, 18.0]))


def test_detrend_removes_linear_trend():
    dates = pd.date_range("2019-01-01", periods=6, freq="YS")
    elapsed_years = (dates - dates[0]) / pd.Timedelta(days=365.25)
    measurements = pd.Series(4.0 * elapsed_years + 12.0, index=dates, name="measurement")

    detrended = CovariateAnalyser.detrend(measurements)

    np.testing.assert_allclose(detrended.to_numpy(), 0.0, atol=1e-8)
    pd.testing.assert_index_equal(detrended.index, dates)
    assert detrended.name == measurements.name


def test_detrend_sorts_unsorted_input():
    dates = pd.date_range("2019-01-01", periods=4, freq="YS")
    elapsed_years = (dates - dates[0]) / pd.Timedelta(days=365.25)
    measurements = pd.Series(
        (4.0 * elapsed_years + 12.0)[[3, 0, 2, 1]],
        index=dates[[3, 0, 2, 1]],
        name="measurement",
    )

    detrended = CovariateAnalyser.detrend(measurements)

    np.testing.assert_allclose(detrended.to_numpy(), 0.0, atol=1e-8)
    pd.testing.assert_index_equal(detrended.index, dates)
    assert detrended.name == measurements.name


def test_fit_power_law_nonlinear():
    x = np.linspace(1.0, 10.0, 10)
    y = 2.5 * x**1.75

    result = CovariateAnalyser.fit_power_law_nonlinear(x=x, y=y)

    assert result["a"] == pytest.approx(2.5)
    assert result["b"] == pytest.approx(1.75)
    assert result["rss"] == pytest.approx(0.0, abs=1e-10)
    assert result["r2"] == pytest.approx(1.0)
    assert result["n"] == len(x)
    assert result["a_ci"][0] <= result["a"] <= result["a_ci"][1]
    assert result["b_ci"][0] <= result["b"] <= result["b_ci"][1]


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (np.array([-1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "x must be >= 0"),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "need more than 2 points"),
    ],
)
def test_fit_power_law_nonlinear_rejects_invalid_data(x, y, message):
    with pytest.raises(ValueError, match=message):
        CovariateAnalyser.fit_power_law_nonlinear(x=x, y=y)


def test_fit_power_law_model_filters_nonpositive_bins(fixture_series_x, fixture_series_y, monkeypatch):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[fixture_series_x, fixture_series_y],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )
    analyser = CovariateAnalyser(timeseries_container=timeseries_container)
    analyser._timeseries_altered = {
        fixture_series_x.label: fixture_series_x,
        fixture_series_y.label: fixture_series_y,
    }
    binned_x = np.array([-1.0, 0.0, 2.0, 4.0])
    binned_y = np.array([10.0, 20.0, 30.0, 40.0])
    fit_result = {"a": 3.0, "b": 1.5}
    fit_inputs = {}

    def align_timeseries(timeseries_label_1, timeseries_label_2, max_lag):
        assert timeseries_label_1 == fixture_series_x.label
        assert timeseries_label_2 == fixture_series_y.label
        assert max_lag == 2
        return 2, AggregationPeriod.MONTH

    def bin_data(x, y, n_bins, method):
        assert x is fixture_series_x.measurements
        assert y is fixture_series_y.measurements
        assert n_bins == 4
        assert method == "quantile"
        return binned_x, binned_y, np.ones(len(binned_x))

    monkeypatch.setattr(analyser, "_align_timeseries", align_timeseries)
    monkeypatch.setattr(analyser, "bin_data", bin_data)

    def fit_power_law(x, y):
        fit_inputs["x"] = x
        fit_inputs["y"] = y
        return fit_result

    monkeypatch.setattr(analyser, "fit_power_law_nonlinear", fit_power_law)

    result, lag = analyser.fit_power_law_model(
        timeseries_label_1=fixture_series_x.label,
        timeseries_label_2=fixture_series_y.label,
        max_lag=2,
        n_bins=4,
        method="quantile",
    )

    assert result == fit_result
    assert lag == 2
    np.testing.assert_array_equal(fit_inputs["x"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(fit_inputs["y"], np.array([30.0, 40.0]))


def test_predict_power_law():
    x = np.array([0.0, 1.0, 2.0])
    model = {"a": 3.0, "b": 2.0}

    result = CovariateAnalyser.predict_power_law(x=x, model=model)

    np.testing.assert_allclose(result, np.array([0.0, 3.0, 12.0]))


def test_predict_power_law_rejects_negative_x():
    with pytest.raises(ValueError, match="x must be >= 0"):
        CovariateAnalyser.predict_power_law(x=np.array([-1.0, 1.0]), model={"a": 2.0, "b": 1.5})
