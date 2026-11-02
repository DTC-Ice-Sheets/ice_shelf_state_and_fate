Overview
========

Ice shelves are permanent extensions of ice sheets that float on the ocean and play a crucial role in regulating ice
sheet dynamics, predominantly around the periphery of the Antarctic Ice Sheet. Because ice shelves are more exposed to
both the air above and the ocean below, they tend to respond faster to temperature increases. Ice shelves do not
directly contribute to sea level rise as they melt because they are already floating on the ocean. However, they
buttress glacial ice flow, reducing the speed at which glacial ice flows into the ocean. As they melt, this buttressing
effect weakens, and the grounded ice accelerates and flows into the ocean.

The **Ice Shelves State and Fate** module of DTC Ice Sheets lets users explore and gain key insights into the Antarctic
ice shelves by combining and interrogating EO, in-situ, and model reanalysis variables across the ice, ocean, and
atmosphere.

This module supports the following primary workflow:

* The univariate and multivariate empirical analysis of ice shelf domain variables to test and discover ice shelf process relationships driving change.

Scientific Background
---------------------
DTC Ice Sheets uses the Ice Shelves State and Fate module to convert heterogeneous input variables across the ice shelf
domain into a homogeneous set of time series, enabling investigation of empirical ice shelf processes through robust
univariate and multivariate statistical analyses.

* The Spatiotemporal Insights module converts input variables in different forms, including gridded, point, and time series datasets, into a standardised time series format with associated uncertainties, where available, calculated using quadrature. Summary statistics of the final time series are available, summarising important statistical and coverage-related quantities. Change detection algorithms are automatically applied to each time series, giving important insights into when large anomalous events or changes occur.
* The Time Series Pre-processor module takes the data variables in the analysis and standardises them across the inputs. This includes ensuring the variables cover the same temporal range and frequency and that missing values are imputed.
* The Covariate Analyser module takes the standardised inputs from the time series pre-processor module and quantifies the strength of the temporal correlations between them using correlation, cross-correlation, and Granger causality. Outputs of this module include statistical plots, correlation metrics and process models.

For an example of a covariate analysis being used to gain insights into ice shelf processes, please see: Gourmelen, N., Jakob, L., Holland, P. R., Dutrieux, P., Goldberg, D., Bevan, S., Luckman, A., & Malczyk, G. (2025). The influence of subglacial lake discharge on Thwaites Glacier ice-shelf melting and grounding-line retreat. Nature Communications, 16(1), 2272.

Repository Structure
--------------------
The primary functionality is exposed through three entry points:

``ice_shelf_state_and_fate.timeseries_module_entrypoint``
    Computes a standardised time series with uncertainties for a given ice shelf and input variable. Univariate summary statistics and change detection functionality are also included.

``ice_shelf_state_and_fate.timeseries_preprocessor_module_entrypoint``
    Homogenises a set of input time series to the same temporal range and frequency, and imputes missing values.

``ice_shelf_state_and_fate.covariate_analyser_module_entrypoint``
    Calculates correlation and causality between input variables. When the analysis includes subglacial lake discharge and basal melt rate variables, it also provides the power-law relationship and associated model.

Data Flow
---------
The Ice Shelf State and Fate workflow consists of three processing steps:

1. Process the input datasets into time series, calculate summary statistics, and apply change-detection algorithms.
2. Prepare the input time series for covariate analysis.
3. Conduct covariate analysis and fit appropriate process models to understand the relationships between ice shelf variables.

You can analyse the results in the DTC dashboard or through the prepared Jupyter notebook.

Integration with the DTC Ice Sheets Dashboard
---------------------------------------------
The Ice Shelves tab of the DTC Ice Sheets dashboard serves as a graphical client for the functionality provided by this
repository.

When a user conducts an analysis in the advanced analysis tab of the ice shelves dashboard, the dashboard invokes the
repository's backend workflows using the same processing pipeline described above.