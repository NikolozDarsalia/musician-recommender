from abc import ABC, abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def load_data(self, paths_dict: dict):
        """
        paths_dict example - {'interactions' : 'user_artists.dat',
                              'items_meta' : 'artists_spotify_matched.parquet'}
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def add_step(self, step, position=None):
        raise NotImplementedError

    @abstractmethod
    def remove_step(self, step_name):
        raise NotImplementedError

    @abstractmethod
    def replace_step(self, step_name, new_step):
        raise NotImplementedError
