import pandas
from pandas.testing import assert_frame_equal

from allocator import allocate_by_ad_group


def test_allocate_by_ad_group():
    manual_data = pandas.DataFrame(
        data={
            'adGroupName': ['a', 'a', 'b'],
            'adGroupId': [1, 1, 2],
        }
    )
    auto_data = pandas.DataFrame(
        data={
            'selection': ['selection1', 'selection2'],
            'adGroupName': ['a', 'c']
        }
    )

    actual_allocation, actual_remainder = allocate_by_ad_group(
        manual_data=manual_data,
        auto_data=auto_data
    )
    expected_allocation = pandas.DataFrame(
        data={
            'suggestion': [1],
            'selection': ['selection1'],
            'feature': ['Allocated via ad group name.']
        }
    )
    assert_frame_equal(actual_allocation, expected_allocation, check_like=True)

    expected_remainder = pandas.DataFrame(
        data={
            'selection': ['selection2'],
            'adGroupName': ['c']
        }
    )
    assert_frame_equal(
        actual_remainder.reset_index(drop=True),
        expected_remainder
    )


def test_allocate_by_ad_group_empty():
    manual_data = pandas.DataFrame(
        columns=['adGroupId', 'adGroupName']
    )
    auto_data = pandas.DataFrame(
        columns=['selection', 'adGroupName']
    )
    actual_allocation, actual_remainder = allocate_by_ad_group(
        manual_data=manual_data,
        auto_data=auto_data
    )
    assert_frame_equal(actual_remainder, auto_data)
