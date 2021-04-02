import pandas
from pandas.testing import assert_frame_equal

from allocator import collect_manual_features


def test_collect_manual_features():
    manual_data = pandas.DataFrame(
        data={
            'adGroupId': [0, 0, 1, 1, 2],
            'adGroupName': ['a', 'a', 'b', 'b', 'c'],
            'keywordText': ['text', 'query', 'text', 'car', 'bra']
        }
    )

    # all the extra little transformations are there because
    # assert_frame_equal is profoundly broken in 0.23.4
    actual = (
        collect_manual_features(
            manual_data=manual_data,
            label_column='adGroupId',
            feature_columns={'adGroupName', 'keywordText'}
        )
        .sort_values('label')
        .sort_values('feature', kind='mergesort')  # mergersort is stable
        .reset_index(drop=True)
    )
    expected = (
        pandas.DataFrame(
            data={
                'feature': [
                    'a', 'text', 'query', 'b',
                    'text', 'car', 'c', 'bra'
                ],
                'label': 3 * [0] + 3 * [1] + 2 * [2]
            }
        )
        .sort_values('label')
        .sort_values('feature', kind='mergesort')
        .reset_index(drop=True)
    )
    assert len(actual.columns) == 2
    actual = actual[expected.columns]
    assert_frame_equal(actual, expected)


def test_collect_manual_features_empty():
    manual_data = pandas.DataFrame(
        columns=['adGroupId', 'adGroupName', 'keywordText']
    )
    actual = collect_manual_features(
        manual_data=manual_data,
        label_column='adGroupId',
        feature_columns={'adGroupName', 'keywordText'}
    )
    expected = pandas.DataFrame(columns=['label', 'feature'])
    assert_frame_equal(actual, expected, check_like=True)
