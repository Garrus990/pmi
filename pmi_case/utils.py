import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime

MIN_OBSERVATIONS_IN_CLASS = 7
# there need to be at least 120 not-na values in a column to consider it as a column that can be analysed
# it is 50% of the data size + 1
N_NOT_NULL = 120
WEEKDAY_TRANSLATION_DAY = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}


def drop_columns_with_single_value(df, verbose=1):
    '''Drop all columns in the data frame df that have only single value - they are uniformative 
    from the modeling point of view, anyhow.

    Args:
        df: data frame to clean of single-value columns
        verbose: whether print info to the user on the number of columns dropped

    Returns:
        pd.DataFrame: data frame without single-value columns
    '''
    df = df.copy()
    num_columns_init = df.shape[1]
    cols_with_multiple_values = df.apply(lambda x: x.unique().shape[0], axis=0)!=1
    df = df.loc[:, cols_with_multiple_values]
    if verbose:
        print(f'{num_columns_init - df.shape[1]} single-value columns dropped.')
    return df


def validate_column(vals, n_not_null=N_NOT_NULL, min_observations_in_class=MIN_OBSERVATIONS_IN_CLASS):
    '''The function checks whether a columns can be modeled using ML models. It also classifies column into
    type of applicable models - binary, multiclass, regression or unknown.

    Args:
        vals: column to validate
        n_not_null: columns needs to have at least n_not_null values in order to be valid
        min_observations_in_class: the function will preserve only groups that have at least 
            `min_observations_in_class` observations

    Returns:
        boolean, str: assessment whether column is valid and either its classification or reason 
            for not being a valid column
    '''
    valid, reason = True, ''
    vals_not_na = vals.dropna()
    if vals_not_na.shape[0] < n_not_null:
        valid, reason = False, f'Not enough values ({vals_not_na.shape[0]}) provided, required: {n_not_null}'
        return valid, reason
    if is_datetime(vals_not_na):
        valid, reason = True, 'datetime'
        return valid, reason

    # determine column type
    n_unique = vals_not_na.unique().shape[0]
    if n_unique <= 1:
        valid, reason = False, 'Not enough classes (0 or 1).'
    elif n_unique > vals_not_na.shape[0]*0.9 and (vals_not_na.dtype != 'float64' and vals_not_na.dtype != 'int64'):
        valid, reason = True, 'identifier'
    elif n_unique == 2:
        valid, reason = True, f'binary'
    # 17 is somewhat arbitrary
    elif n_unique > 2 and ((n_unique < 17 and vals_not_na.dtype == 'float64') or \
                           (vals_not_na.dtype != 'float64' and vals_not_na.dtype != 'int64')):
        valid, reason = True, 'multiclass'
    elif n_unique > 2 and (vals_not_na.dtype == 'float64' or vals_not_na.dtype == 'int64'):
        valid, reason = True, 'regression'
    else:
        valid, reason = False, 'Type not known.'

    # second pass - this time we drop small groups for binary and multiclass variables
    # and check whether the type changed
    if reason == 'binary' or reason == 'multiclass':
        vals_not_na = drop_infrequent_groups(vals_not_na, min_observations_in_class)
        n_unique = vals_not_na.unique().shape[0]
        if n_unique <= 1:
            valid, reason = False, 'Not enough classes (0 or 1).'
        elif n_unique == 2:
            valid, reason = True, 'binary'
        else:  # otherwise the type - multiclass is preserved
            pass

    return valid, reason


def drop_infrequent_groups(vals, min_observations_in_class=MIN_OBSERVATIONS_IN_CLASS):
    '''For vals, count the occurrence of unique values in groups and, if the count is below
    min_observations_in_class - replace these values with pd.NA.

    Args:
        vals: group values
        min_observations_in_class: threshold of 'infrequency'
    Returns:
        pd.Series: series with infrequent groups replaced by NAs
    '''
    type_ = vals.dtype
    vc = vals.value_counts()
    infrequent_entries = vc.loc[vc < min_observations_in_class].index.tolist()
    vals = vals.replace(infrequent_entries, pd.NA).dropna().astype(type_)  # to preserve the type
    return vals


def drop_erroneous_values(vals, quantile=0.01, num_deviations=8, verbose=True):
    '''The function is devised to trim (most likely) erroneous values from a variable.
    A value is deemed to be erroneous when it is `num_deviations` from the trimmed mean
    of the provided values. Trimming is quantile-based, i.e. top and bottom `quantile` percent
    of observations are removed before both mean and standard deviation are calculated.

    Args:
        vals: iterable with values to inspect
        quantile: what percentage of top and bottom observations to trim
        num_deviations: an observation is deemed erroneous whenever it is at least 
            num_deviations from the trimmed mean
        verbose: whether to print an info about the number of dropped values
    '''
    vals = vals.dropna()
    idx = vals.index
    name = vals.name
    # drop top and bottom quantile percent of observations
    trimmed = vals.loc[(vals > vals.quantile(quantile)) & (vals < vals.quantile(1-quantile))]
    m, std = trimmed.mean(), trimmed.std()
    # 8 deviations (default) is arbitrarily chosen
    ll = m - num_deviations*std
    ul = m + num_deviations*std
    # values outside of the interval are replaced with NaNs
    # they cannot be replaced with pd.NA because of issues later on, in the MissForest
    vals = pd.Series(np.where(np.logical_or(vals < ll, vals > ul), np.nan, vals),
                     index=idx, name=name)
    if verbose:
        print(f'In the column {vals.name}, {vals.isna().sum()} values were replaced.')
    return vals