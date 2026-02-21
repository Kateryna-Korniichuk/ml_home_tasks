"""process_bank_churn.py

Preprocessing helpers for Kaggle competition:
bank-customer-churn-prediction-dlu-course-c-4
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_feature_columns(raw_df: pd.DataFrame, *, target_col: str, drop_cols: List[str]) -> List[str]:
    """Select raw feature columns excluding target and drop columns."""
    return [c for c in raw_df.columns if c != target_col and c not in drop_cols]


def split_train_val(
    raw_df: pd.DataFrame,
    *,
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split dataframe into train and validation sets with stratification."""
    X = raw_df.drop(columns=[target_col]).copy()
    y = raw_df[target_col].copy()

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_encoder(
    X_train_raw: pd.DataFrame,
    *,
    categorical_cols: List[str],
) -> Tuple[OneHotEncoder, List[str]]:
    """Fit OneHotEncoder on categorical columns."""
    encoder = OneHotEncoder(
        sparse_output=False,
        drop="first",
        handle_unknown="ignore",
    )
    encoder.fit(X_train_raw[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    return encoder, encoded_cols


def encode_categorical(
    X_raw: pd.DataFrame,
    *,
    encoder: OneHotEncoder,
    categorical_cols: List[str],
    encoded_cols: List[str],
) -> pd.DataFrame:
    """Apply one-hot encoding and drop original categorical columns."""
    X = X_raw.copy()
    encoded = encoder.transform(X[categorical_cols])
    X[encoded_cols] = encoded
    return X.drop(columns=categorical_cols)


def fit_scaler(
    X_train: pd.DataFrame,
    *,
    numeric_cols: List[str],
) -> StandardScaler:
    """Fit StandardScaler on numeric columns."""
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])
    return scaler


def scale_numeric(
    X: pd.DataFrame,
    *,
    scaler: StandardScaler,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Scale numeric columns using fitted scaler."""
    X_out = X.copy()
    X_out.loc[:, numeric_cols] = scaler.transform(X_out[numeric_cols])
    return X_out


def preprocess_data(
    raw_df: pd.DataFrame,
    *,
    target_col: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 12,
    scale_numeric_flag: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder,
]:
    """Split and preprocess training data."""
    drop_cols = ["id", "CustomerId", "Surname"]
    categorical_cols = ["Geography", "Gender"]
    numeric_cols_to_scale = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]

    raw_input_cols = select_feature_columns(
        raw_df,
        target_col=target_col,
        drop_cols=drop_cols,
    )

    X_train_raw, X_val_raw, y_train, y_val = split_train_val(
        raw_df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_raw = X_train_raw[raw_input_cols].copy()
    X_val_raw = X_val_raw[raw_input_cols].copy()

    encoder, encoded_cols = fit_encoder(
        X_train_raw,
        categorical_cols=categorical_cols,
    )

    X_train = encode_categorical(
        X_train_raw,
        encoder=encoder,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )
    X_val = encode_categorical(
        X_val_raw,
        encoder=encoder,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )

    scaler: Optional[StandardScaler] = None
    if scale_numeric_flag:
        scaler = fit_scaler(
            X_train,
            numeric_cols=numeric_cols_to_scale,
        )
        X_train = scale_numeric(
            X_train,
            scaler=scaler,
            numeric_cols=numeric_cols_to_scale,
        )
        X_val = scale_numeric(
            X_val,
            scaler=scaler,
            numeric_cols=numeric_cols_to_scale,
        )

    input_cols = list(X_train.columns)
    X_val = X_val.reindex(columns=input_cols)

    encoder._raw_input_cols = raw_input_cols  # type: ignore
    encoder._categorical_cols = categorical_cols  # type: ignore
    encoder._encoded_cols = encoded_cols  # type: ignore
    encoder._numeric_cols_to_scale = numeric_cols_to_scale  # type: ignore

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(
    raw_df: pd.DataFrame,
    *,
    input_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[StandardScaler],
    scale_numeric_flag: bool = True,
) -> pd.DataFrame:
    """Preprocess new/unseen data using fitted encoder and scaler."""
    raw_input_cols = encoder._raw_input_cols  # type: ignore
    categorical_cols = encoder._categorical_cols  # type: ignore
    encoded_cols = encoder._encoded_cols  # type: ignore
    numeric_cols_to_scale = encoder._numeric_cols_to_scale  # type: ignore

    X_raw = raw_df[raw_input_cols].copy()

    X = encode_categorical(
        X_raw,
        encoder=encoder,
        categorical_cols=categorical_cols,
        encoded_cols=encoded_cols,
    )

    if scale_numeric_flag:
        if scaler is None:
            raise ValueError("scaler is None but scale_numeric_flag=True")
        X = scale_numeric(
            X,
            scaler=scaler,
            numeric_cols=numeric_cols_to_scale,
        )

    return X.reindex(columns=input_cols)
