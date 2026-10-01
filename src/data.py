"""Loading, splitting and preprocessing the credit card dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\nDownload it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in the data/ folder (see README, section 'Dataset')."
        )
    df = pd.read_csv(path)
    missing = set(config.FEATURE_COLUMNS + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    # The raw file contains ~1,000 exact duplicate rows. Duplicates can land in
    # both train and test and inflate scores (data leakage), so we drop them.
    return df.drop_duplicates().reset_index(drop=True)


def get_split_frames(df: pd.DataFrame = None):
    """Stratified train/val/test split (raw, unscaled).

    Stratify = keep the same fraud percentage in every split. Without it, a
    random split of a 0.17 % class could leave the test set with almost no fraud.
    Returns a dict with keys X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if df is None:
        df = load_data()
    X, y = df[config.FEATURE_COLUMNS], df[config.TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=config.TEMP_SIZE, stratify=y, random_state=config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=config.TEST_SIZE_OF_TEMP, stratify=y_tmp, random_state=config.SEED)
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test)


def build_preprocessor() -> ColumnTransformer:
    """Standardise Time and Amount (fit on TRAIN only); pass V1..V28 through.

    Output stays a DataFrame so SHAP plots can show real feature names.
    """
    return ColumnTransformer(
        [("scale", StandardScaler(), config.COLUMNS_TO_SCALE)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
