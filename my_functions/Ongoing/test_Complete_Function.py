import pandas as pd
import pytest
from my_functions.Ongoing.Complete_Function import Extract_Diff_id

def test_extract_diff_id_basic_filtering():
    # Happy path: some items are filtered out
    tota_FB_promotion = pd.DataFrame({
        'promo_cat': [1, 2, 3, 4],
        'value': ['a', 'b', 'c', 'd']
    })

    Question_user = pd.DataFrame({
        'user_var': [2, 4, 5],
        'other': ['x', 'y', 'z']
    })

    result = Extract_Diff_id('promo_cat', 'user_var', tota_FB_promotion, Question_user)

    # Should only contain 1 and 3 because 2 and 4 are in Question_user
    expected_ids = [1, 3]
    actual_ids = result['promo_cat'].astype(int).tolist()

    assert actual_ids == expected_ids
    assert len(result) == 2

def test_extract_diff_id_no_overlap():
    # No overlap: none are filtered out
    tota_FB_promotion = pd.DataFrame({
        'promo_cat': [1, 2, 3],
        'value': ['a', 'b', 'c']
    })

    Question_user = pd.DataFrame({
        'user_var': [4, 5, 6],
        'other': ['x', 'y', 'z']
    })

    result = Extract_Diff_id('promo_cat', 'user_var', tota_FB_promotion, Question_user)

    expected_ids = [1, 2, 3]
    actual_ids = result['promo_cat'].astype(int).tolist()

    assert actual_ids == expected_ids
    assert len(result) == 3

def test_extract_diff_id_full_overlap():
    # Full overlap: all are filtered out
    tota_FB_promotion = pd.DataFrame({
        'promo_cat': [1, 2, 3],
        'value': ['a', 'b', 'c']
    })

    Question_user = pd.DataFrame({
        'user_var': [1, 2, 3, 4],
        'other': ['x', 'y', 'z', 'w']
    })

    result = Extract_Diff_id('promo_cat', 'user_var', tota_FB_promotion, Question_user)

    assert len(result) == 0

def test_extract_diff_id_no_mutation():
    # Ensure original dataframes are not mutated
    tota_FB_promotion = pd.DataFrame({
        'promo_cat': [1, 2, 3],
        'value': ['a', 'b', 'c']
    })

    Question_user = pd.DataFrame({
        'user_var': [2, 4],
        'other': ['x', 'y']
    })

    tota_FB_promotion_copy = tota_FB_promotion.copy()
    Question_user_copy = Question_user.copy()

    Extract_Diff_id('promo_cat', 'user_var', tota_FB_promotion, Question_user)

    pd.testing.assert_frame_equal(tota_FB_promotion, tota_FB_promotion_copy)
    pd.testing.assert_frame_equal(Question_user, Question_user_copy)
