from tsds.tsds import TimeSeriesDataSegmentation

__author__ = "J.Udovic"
__email__ = "jakobudovic2@gmail.com"
__version__ = "0.0.2"

__doc__ = """
TimeSeriesDataSegmentation
================

Description
-----------
TimeSeriesDataSegmentation is a Python package created for analysis of any (clinical) time series data.

Example
-------
>>> # Import library
>>> from tsds import TimeSeriesDataSegmentation
>>> # Initialize with 2 datasets
>>> tds = TimeSeriesDataSegmentation(df_clustering, df_prediction)
>>> # Reduce data dimensionality using NMF
>>> df_clustering_nmf = tds.get_nmf_data(
    ids_col=ids_col,
    timestamp_col=timestamp_col,
)

References
----------
* https://github.com/jakobudovic/tsds

"""
