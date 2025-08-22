"""Test the covariate analyser module."""

import numpy as np
import pandas as pd
import pytest

from use_case_3.covariate_analyser_module.covariate_analyser_module import (
    CovariateAnalyser,
    ModelEvaluationMetric,
    JointPlotAnalysisType,
)
from use_case_3.timeseries_module.timeseries import Timeseries, TimeseriesType

from use_case_3.covariate_analyser_module.timeseries_preprocessor import (
    AggregationPeriod,
    AggregationStrategy,
    TimeSeriesPreprocessor,
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
        measurements=timeseries, label="time series 1", unit="unit 1", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 2", unit="unit 2", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 3", unit="unit 3", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 4", unit="unit 4", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 5", unit="unit 5", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 6", unit="unit 6", timeseries_type=TimeseriesType("cumulative")
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
        measurements=timeseries, label="time series 7", unit="unit 7", timeseries_type=TimeseriesType("cumulative")
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
        timeseries_type=TimeseriesType("cumulative"),
    )


def test_correlation_coefficient(timeseries_1_fixture, timeseries_2_fixture, timeseries_7_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    timeseries_container._timeseries_altered[timeseries_2_fixture.label] = timeseries_7_fixture

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
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


def test_covariance(timeseries_1_fixture, timeseries_1_inv_fixture, timeseries_2_fixture, timeseries_7_fixture):
    timeseries_container = TimeSeriesPreprocessor(
        timeseries=[timeseries_1_fixture, timeseries_2_fixture, timeseries_1_inv_fixture, timeseries_7_fixture],
        time_period=AggregationPeriod.MONTH,
        aggregation=AggregationStrategy.MEAN,
        process_timeseries=False,
    )

    timeseries_container._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    timeseries_container._timeseries_altered[timeseries_7_fixture.label] = timeseries_7_fixture

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)
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

    timeseries_container._timeseries_altered[timeseries_1_fixture.label] = timeseries_1_fixture
    timeseries_container._timeseries_altered[timeseries_1_inv_fixture.label] = timeseries_1_inv_fixture

    covariate_analyser = CovariateAnalyser(timeseries_container=timeseries_container)

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


def test__align_timeseries(timeseries_7_fixture, timeseries_7_lag_fixture):
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

    freq, corr_lag = covariate_analyser._align_timeseries(
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
        timeseries_container._timeseries_altered[timeseries_7_fixture.label].measurements.values, timeseries_1_gt.values
    )
    assert np.allclose(
        timeseries_container._timeseries_altered[timeseries_7_lag_fixture.label].measurements.values,
        timeseries_2_gt.values,
    )
    assert freq == "M"
    assert corr_lag == -1


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
