# tsds: Time Series Data Segmentation Algorithm

[![Python](https://img.shields.io/pypi/pyversions/tsds)](https://img.shields.io/pypi/pyversions/tsds)
[![Pypi](https://img.shields.io/pypi/v/tsds)](https://pypi.org/project/tsds/)
[![LOC](https://sloc.xyz/github/jakobudovic/tsds/?category=code)](https://github.com/jakobudovic/tsds/)
[![Downloads](https://static.pepy.tech/personalized-badge/tsds?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/tsds)
[![Downloads](https://static.pepy.tech/personalized-badge/tsds?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/tsds)
[![License](https://img.shields.io/badge/license-GNU-green.svg)](https://github.com/jakobudovic/tsds/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/jakobudovic/tsds.svg)](https://github.com/jakobudovic/tsds/network)
[![Issues](https://img.shields.io/github/issues/jakobudovic/tsds.svg)](https://github.com/jakobudovic/tsds/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

This is a Python library for time series data segmentation, specifically developed for clinical data. It includes the following components:

1. Dimensionality reduction using Non-negative Matrix Factorization (NMF)
2. Optimal number of clusters calculation using Silhouette score, Calinski Harabasz score, and Davies Bouldin score.
3. Predictive modeling using Multilayer Perceptron (MLP) classifier, Support Vector Machines (SVM), and Random Forest.
4. Explanation of cluster groups using SHAP values.
5. Analysis and simulation of disease progression using skip grams and Markov chains, with visual representation of group likelihood changes.

## Usage

To use the library, simply import it into your project and follow the steps outlined in the components above. Detailed usage instructions and examples can be found in the library's documentation.

## Dependencies

The library requires the following dependencies:

- NumPy
- Pandas
- Scikit-learn
- SHAP
- Matplotlib (for visual representation)

## Contribution

We welcome contributions to this library. If you have any suggestions or bug reports, please create a GitHub issue. If you would like to contribute code, please submit a pull request.

## License

This library is available under the GNU General Public License Version 3.
