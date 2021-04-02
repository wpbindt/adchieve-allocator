from typing import Dict, List, Set, Tuple

from nltk import SnowballStemmer
import numpy
import pandas
try:
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
except ImportError:
    print('Installing sklearn')
    # import subprocess
    # import sys
    # subprocess.check_call(
    #     [sys.executable, '-m', 'pip', 'install', 'sklearn']
    # )

STEMMER = SnowballStemmer(language='english')

MANUAL_FEATURE_COLUMNS = {
    'adGroupName',
    'searchTerm',
    'campaignName',
    'keywordText'
}

AUTO_FEATURE_COLUMNS = [
    'selection',
    'searchTerm',
    'campaignName',
    'adGroupName',
]


def check_column_existence(df: pandas.DataFrame, columns: Set[str]):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f'Missing column: {column}')


def allocate_by_ad_group(
    auto_data: pandas.DataFrame,
    manual_data: pandas.DataFrame
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    allocations = (
        auto_data
        [['selection', 'adGroupName']]
        .merge(
            manual_data[['adGroupName', 'adGroupId']],
            on=['adGroupName']
        )
        .drop('adGroupName', axis=1)
        .rename(columns={'adGroupId': 'suggestion'})
        .assign(feature='Allocated via ad group name.')
        .drop_duplicates()
    )
    remainder = auto_data[
        ~auto_data.selection.isin(allocations.selection.unique())
    ]
    return allocations, remainder


def collect_manual_features(
    manual_data: pandas.DataFrame,
    label_column: str,
    feature_columns: Set[str]
) -> pandas.DataFrame:
    """
    Put all features into one column, vertically. Output is dataframe with
    columns `label` and `feature`.
    """
    reindexed_df = manual_data.set_index(label_column)
    return (
        pandas.concat(
            [
                reindexed_df[column].rename('feature')
                for column in feature_columns
            ]
        )
        .reset_index()
        .drop_duplicates()
        .rename(columns={label_column: 'label'})
    )


def collect_auto_features(
    auto_data: pandas.DataFrame,
    id_column: str,
    feature_columns: List[str],
    attribute_columns: List[str]
) -> pandas.DataFrame:
    """
    Concatenate all features horizontally, separated by a space.
    Output is dataframe with columns id_column, attribute_columns,
    and `feature`.
    """
    return (
        auto_data
        .assign(
            feature=auto_data[feature_columns].agg(' '.join, axis=1)
        )
        [['feature', id_column] + attribute_columns]
    )


def stem(phrase: str) -> str:
    return ' '.join(
        STEMMER.stem(word)
        for word in phrase.split()
    )


def fit_model(
    model: sklearn.pipeline.Pipeline,
    manual_data: pandas.DataFrame,
    param_grid: Dict[str, object]
) -> sklearn.pipeline.Pipeline:
    grid_search = GridSearchCV(
        model,
        param_grid,
        n_jobs=-1
    )
    grid_search.fit(manual_data.feature, manual_data.label)
    output_model = grid_search.best_estimator_
    output_model.fit(manual_data.feature, manual_data.label)
    return output_model


def main(
    manual_data: pandas.DataFrame,
    auto_data: pandas.DataFrame
) -> pandas.DataFrame:
    allocations, remaining_auto_data = allocate_by_ad_group(
        auto_data=auto_data,
        manual_data=manual_data
    )

    clean_manuals = collect_manual_features(
        manual_data=manual_data,
        label_column='adGroupId',
        feature_columns=MANUAL_FEATURE_COLUMNS
    )

    vec = CountVectorizer()
    classifier = MultinomialNB()
    model = make_pipeline(vec, classifier)
    param_grid = {
        'countvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'countvectorizer__preprocessor': [stem],
        'countvectorizer__strip_accents': [None, 'ascii', 'unicode'],
        'multinomialnb__alpha': numpy.linspace(0, 1, num=8),
        'multinomialnb__fit_prior': [False]
    }

    fitted_model = fit_model(
        model=model,
        manual_data=clean_manuals,
        param_grid=param_grid
    )

    clean_autos = collect_auto_features(
        auto_data=remaining_auto_data,
        feature_columns=AUTO_FEATURE_COLUMNS,
        id_column='selection',
        attribute_columns=['selectionType']
    )

    ml_allocations = (
        clean_autos
        .assign(suggestion=fitted_model.predict(clean_autos.feature))
    )

    allocations = pandas.concat([allocations, ml_allocations])

    pre_existing_keywords = manual_data.keywordText.unique()
    return allocations[~allocations.selection.isin(pre_existing_keywords)]
