"""Classical TF-IDF + Logistic Regression baseline for comparison."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from .config import RANDOM_STATE


def build_baseline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=2000,
                                                random_state=RANDOM_STATE))])
