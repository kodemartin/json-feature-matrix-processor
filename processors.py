import json
import numpy as np
import pandas as pd
import pdb

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class JsonData:
    """Represent JSON data read from a file, providing
    a series of preprocessing methods.

    A fixed structure of the JSON data is presumed, conforming
    with the sample in ``./data.json``.

    :param str filename: The name of the json file
    """

    def __init__(self, filename):
        self.filename = filename
        self.pca = None     # Points to the most-recently used
                            # PCA model
        self.scaler = None  # Points to the most-recently used
                            # scaler
        # Cached properties
        self._raw = None
        self._df = None
        self._vector_features = None
        self._h_df = None
        self._stats = None
        self._fences = None
        self._filled_df = None
        self._outliers = None
        self._reduced_df = None

    @property
    def raw(self):
        """Get the raw JSON data either from encapsulated
        cache, or by loading it from storage.

        :rtype: dict
        """
        if self._raw is None:
            with open(self.filename) as f:
                self._raw = json.loads(f.read())
        return self._raw

    @property
    def df(self):
        """Get the dataframe representation of the JSON data
        either from the encapsulated cache, or by doing
        the conversion first.

        :rtype: `pandas.DataFrame`
        """
        if self._df is None:
            self._df = self.to_df()
        return self._df

    @property
    def vector_features(self):
        """Get a list of the names of multivalued
        features, as referred to in the dataframe
        representation of the data.

        :rtype: list
        """
        if self._vector_features is None:
            self._vector_features = list(self.get_vector_feature_names())
        return self._vector_features

    @property
    def homogeneous_df(self):
        """Get an homogenized version of the dataframe
        representation of the data, where all vector features
        are reduced to their norm.

        Use cache if available.

        :rtype: `pandas.DataFrame`
        """
        if self._h_df is None:
            self._h_df = self.transform_vector_features()
        return self._h_df

    @property
    def stats(self):
        """Get the basic statistics of each features
        across the dataset.

        Use cache if available.

        :rtype: `pandas.DataFrame`
        """
        if self._stats is None:
            self._stats = self.homogeneous_df.describe()
        return self._stats

    @property
    def fences(self):
        """Get the inner and outer fences of the feature values
        with reference to the homogeneous dataframe.

        Uses the cache if available, otherwise invokes
        the `self.evaluate_fences` method.

        :rtype: dict
        :return: A map ``{<fence-type>: pandas.DataFrame}``
            where each dataframe has two rows with the
            lower and upper bounds of each type of fence
            for all features.
        """
        if self._fences is None:
            self._fences = self.evaluate_fences()
        return self._fences

    @property
    def outliers(self):
        """Get the outliers for each feature. Use
        cache if available, otherwise evaluate them.

        We consider as outliers all values outside the so-called
        inner fence of each feature (see `self.fences`).

        :rtype: `pandas.DataFrame`
        """
        if self._outliers is None:
            self._outliers = self.evaluate_outliers()
        return self._outliers

    @property
    def filled_df(self):
        """Get a transformed version of the homogeneous
        dataframe, where all null values have been filled
        with a value outside the outer fences of the respective
        feature.

        Serves for visualizing the null values as outliers.

        The method uses the cache if available, otherwise
        performs a lazy evaluation.

        :rtype: `pandas.DataFrame`
        """
        if self._filled_df is None:
            self._filled_df = self.fill_homogeneous_df()
        return self._filled_df

    @property
    def reduced_df(self):
        """Get the transformed feature matrix.

        Uses the cache, otherwises applyies dimensionality
        reduction on `self.filled_df`.

        :rtype: `pandas.DataFrame`
        """
        if self._reduced_df is None:
            self._reduced_df = self.reduce_dimensions()
        return self._reduced_df

    def fill_homogeneous_df(self, fence_type='outer'):
        """Fill null values in the homogeneous dataframe
        with values outside the outer fences of the
        respective features.

        :param str fence_type: The type of the fence. By default
            we consider the outer fences.
        :rtype: `pandas.DataFrame`
        """
        fence = self.fences[fence_type]
        return (self.homogeneous_df.replace([np.inf, -np.inf], np.nan)
                                   .apply(lambda s: s.mask(np.isnan,
                                          1.1*fence.loc['upper'][s.name])))

    def evaluate_outliers(self, fence_type='inner'):
        """Evaluate the values that lie outside
        the boundaries of the specified fence type.

        :param str fence_type: The type of the fence. By default
            we consider the inner fences.
        :rtype: `pandas.DataFrame`
        """
        fence = self.fences[fence_type]
        condition = self.filled_df < fence.loc['lower']
        condition |= (self.filled_df > fence.loc['upper'])
        outliers = self.filled_df[condition].dropna(axis=1, how='all')
        return outliers

    def plot_outliers(self, feature=None, save=True):
        """Plot the outliers for the specified feature.

        If no feature is specified, a plot with all features
        that have outliers is generated.

        :param feature: The columns of `self.filled_df` to use
            for the boxplot. It follows the signature of
            `pandas.DataFrame.boxplot`.
        :type feature: str or list or None
        :param bool save: If `True` save the plot into
            a file.
        :rtype: `matplotlib.axes.Axes`
        """
        ax = self.filled_df.boxplot(column=feature or list(self.outliers))
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        if feature is None:
            ax.set_yscale('symlog')
        ax.set_title('Feature outliers')
        if save:
            ax.figure.savefig(self.filename.replace('json', 'png'), dpi=300)
        return ax

    def evaluate_fences(self):
        """"We follow the steps in

            https://www.wikihow.com/Calculate-Outliers

        to evaluate the lower and upper bounds of each type
        of fence.

        :rtype: dict
        :return: A map ``{<fence-type>: pandas.DataFrame}``
            where each dataframe has two rows with the
            lower and upper bounds of each type of fence
            for all features.
        """
        quartiles = self.stats.loc[['25%', '75%']]
        inter_quartile_distance = quartiles.iloc[1] - quartiles.iloc[0]
        # Evaluate inner fence
        inner_fence = quartiles.set_index(pd.Index(['lower', 'upper']))
        inner_fence.iloc[0] = inner_fence.iloc[0] - 1.5*inter_quartile_distance
        inner_fence.iloc[1] = inner_fence.iloc[1] + 1.5*inter_quartile_distance
        # Evaluate outer fence
        outer_fence = quartiles.set_index(pd.Index(['lower', 'upper']))
        outer_fence.iloc[0] = inner_fence.iloc[0] - 3.0*inter_quartile_distance
        outer_fence.iloc[1] = inner_fence.iloc[1] + 3.0*inter_quartile_distance
        return {'inner': inner_fence, 'outer': outer_fence}

    def get_vector_feature_names(self):
        """Return a generator of the names of multivalued
        features. Names are consistent with the dataframe
        representation of the data
        """
        for feature, values in self.df.items():
            try:
                values[0] + 2
            except TypeError:
                # Values are lists
                yield feature

    def to_df(self):
        """Convert the raw json to a dataframe object.

        :rtype: `pandas.DataFrame`
        """
        df = None
        for cluster, instances in self.raw.items():
            data = dict(instances) # We want self.raw to remain intact
            data['cluster'] = cluster
            next_df = pd.io.json.json_normalize(
                data, 'cluster_instances', ['cluster']
                )
            try:
                df = df.append(next_df)
            except AttributeError:
                df = next_df

        new_column_names = {f: f.replace('features.', '') for f in df}
        new_column_names['name'] = 'instance'
        df.rename(columns=new_column_names, inplace=True)
        df.set_index(['cluster', 'instance'], inplace=True)
        return df

    def transform_vector_features(self, transform=None):
        """Transform values of multivalued features in
        the dataframe representation, by applying
        dynamically a function (by default `numpy.linalg.norm`).

        :param tranform: callable or None
        :rtype: `pandas.DataFrame`
        :return: A new dataframe instance.
        """
        transform = transform or np.linalg.norm
        transformed_df = self.df.copy()
        for vector_feature in self.vector_features:
            transformed_df[vector_feature] = self.df[vector_feature].apply(
                transform
                )
        return transformed_df

    def to_csv(self):
        """Save the dataframe representation of the data
        into csv format. The dataframe is first transformed
        so that multivalued features are concatenated into
        a ',' delimited string.

        The ';' delimiter is used for the csv.
        """
        transformed_df = self.transform_vector_features(
            lambda l: ','.join(map(str, l))
            )
        transformed_df.to_csv(self.filename.replace('json', 'csv'), sep=';')

    def reduce_dimensions(self, n_components=10, **kwargs):
        """Reduce the dimensions of `self.filled_df` using principal-component
        analysis (PCA).

        The PCA model is stored in `self.pca`.

        :param int n_components: The number of reduced dimensions.
        :param `**kwargs`: Keyword arguments of `sklearn.decomposition.PCA`.
        :rtype: `pandas.DataFrame`
        :return: The dataframe with the transformed feature matrix.
        """
        self.pca = PCA(n_components=n_components, **kwargs)
        self.scaler = StandardScaler()
        scaled_df = self.scaler.fit_transform(self.filled_df)
        return pd.DataFrame(self.pca.fit_transform(scaled_df),
                            index=self.filled_df.index)


if __name__ == "__main__":
    d = JsonData('./data.json')
    df = d.to_df()
    # d.plot_outliers()
