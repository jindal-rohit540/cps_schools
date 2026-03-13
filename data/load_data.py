"""
data/load_data.py
─────────────────
Kaggle Students Performance in Exams — clean loader with NO data leakage.

Two model scenarios:
  A) PRE-EXAM: demographic features only → predict at_risk  (actionable early warning)
  B) POST-EXAM: use math+reading → predict writing performance (cross-subject)

Source: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
1000 students · 8 raw columns · 0 missing values
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "StudentsPerformance.csv")


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    df = df.rename(columns={
        'gender':                      'gender',
        'race/ethnicity':              'race_ethnicity',
        'parental level of education': 'parent_education',
        'lunch':                       'lunch',
        'test preparation course':     'test_prep',
        'math score':                  'math_score',
        'reading score':               'reading_score',
        'writing score':               'writing_score',
    })

    # ── Score aggregates (used for EDA & target, NOT as input features) ───────
    df['avg_score']   = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
    df['total_score'] = df['math_score']  + df['reading_score'] + df['writing_score']

    # ── Score gaps ────────────────────────────────────────────────────────────
    df['math_vs_verbal'] = df['math_score'] - ((df['reading_score'] + df['writing_score']) / 2)

    # ── Pass/fail per subject ─────────────────────────────────────────────────
    df['pass_math']       = (df['math_score']    >= 60).astype(int)
    df['pass_reading']    = (df['reading_score'] >= 60).astype(int)
    df['pass_writing']    = (df['writing_score'] >= 60).astype(int)
    df['subjects_passed'] = df['pass_math'] + df['pass_reading'] + df['pass_writing']

    # ── DEMOGRAPHIC / CONTEXT features (known BEFORE exam — no leakage) ───────
    df['low_ses']        = (df['lunch'] == 'free/reduced').astype(int)
    df['completed_prep'] = (df['test_prep'] == 'completed').astype(int)
    df['is_female']      = (df['gender'] == 'female').astype(int)

    edu_map = {
        'some high school': 0, 'high school': 1, 'some college': 2,
        "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5,
    }
    df['parent_edu_ord'] = df['parent_education'].map(edu_map)
    df['race_ord']       = df['race_ethnicity'].map({
        'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4
    })

    # ── Performance tier ──────────────────────────────────────────────────────
    df['performance_tier'] = pd.cut(
        df['avg_score'],
        bins=[0, 40, 55, 70, 85, 101],
        labels=['Failing', 'Below Average', 'Average', 'Good', 'Excellent']
    )

    # ── TARGET: at_risk = bottom 25% avg score ────────────────────────────────
    threshold     = df['avg_score'].quantile(0.25)
    df['at_risk'] = (df['avg_score'] < threshold).astype(int)

    return df


# ── SCENARIO A: Pre-exam demographic features only (clean, no leakage) ────────
FEATURE_COLS = [
    'low_ses',          # Free/reduced lunch → SES proxy
    'completed_prep',   # Test preparation course completed
    'parent_edu_ord',   # Parental education (0–5 ordinal)
    'is_female',        # Gender
    'race_ord',         # Race/ethnicity group (0–4 ordinal)
]

# ── SCENARIO B: Use math score to predict avg (post one-exam prediction) ──────
FEATURE_COLS_POST = [
    'low_ses', 'completed_prep', 'parent_edu_ord', 'is_female', 'race_ord',
    'math_score',   # Only math — predicting overall from one subject
]

TARGET_COL = 'at_risk'


if __name__ == '__main__':
    df = load_and_clean()
    print(f"Shape: {df.shape}")
    print(f"At-risk rate: {df['at_risk'].mean()*100:.1f}%")
    print(f"Avg score:    {df['avg_score'].mean():.1f}")
    print(f"\nPerformance tier:")
    print(df['performance_tier'].value_counts().sort_index())
    print(f"\nFeature set A (pre-exam):\n{FEATURE_COLS}")
