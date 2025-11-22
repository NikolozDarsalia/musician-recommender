from interfaces.base_preprocessor import BasePreprocessor


class TimePreprocessor(BasePreprocessor):
    """
    Converts epoch timestamp → {hour, day, week_of_year, recency}
    """
