# tag_aggregator.py
import pandas as pd
from ...interfaces.base_preprocessor import BasePreprocessor


class TagAggregator(BasePreprocessor):
    """
    Generic tag aggregation utility.

    Aggregates tags into a pipe-separated string by a chosen key
    (e.g. userID or artistID).

    Examples:
    - group_by="artistID" → artist-level tags
    - group_by="userID" → user-level tags
    """

    def __init__(
        self,
        group_by: str,
        tag_col: str = "tagValue",
        output_col: str = "aggregated_tags",
    ):
        self.group_by = group_by
        self.tag_col = tag_col
        self.output_col = output_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.dropna(subset=[self.tag_col])
            .groupby(self.group_by)[self.tag_col]
            .apply(lambda x: "|".join(sorted(set(map(str, x)))))
            .reset_index(name=self.output_col)
        )
