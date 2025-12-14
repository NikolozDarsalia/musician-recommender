<<<<<<< HEAD
from recommender_pipeline.interfaces.base_pipeline import BasePipeline
from recommender_pipeline.data_loaders.data_loader import StandardLoader
from data_loaders import artist_match


class RecommenderPipeline(BasePipeline):
    def __init__(
        self,
        steps=None,
        test_size=0.15,
        val_size=0.15,
        interactions_data_pat="../../data/user_artists_dat",
        artists_name_path="../../data/artists.dat",
        artists_audio_path="../../data/spotify_musics.parquet",
        random_state=42,
        levenshtein_score_cutoff=0.85,
    ):
        """
        steps: list of PipelineStep instances
        """
        self.test_size = test_size
        self.val_size = val_size
        self.interactions_data_path = interactions_data_pat
        self.artists_name_path = artists_name_path
        self.artists_audio_path = artists_audio_path

        self.random_state = random_state
        self.levenshtein_score_cutoff = levenshtein_score_cutoff

        if steps == None:
            self.steps = "our_steps"
        else:
            self.steps = steps

        self.results = {}

    def run(self, initial_data=None, until_step=None):
        """
        Executes pipeline up to a specific step (optional)
        """
        (
            train_df,
            val_df,
            test_df,
        ) = self._data_loader()

        for step in self.steps:
            step_name = step.name()

            # run step
            data = step.fit_transform(train_df)

            # store output
            self.results[step_name] = data

            # stop if user requested partial run
            if until_step and step_name == until_step:
                break

        return data

    def _data_loader(self):
        interactions_loader = StandardLoader(self.interactions_data_path)

        interactions_data = interactions_loader.load()

        train_df, val_df, test_df = interactions_loader.train_test_val_split(
            df=interactions_data,
            strategy="user_stratified",
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
        )

        train_df = self._last_fm_spotify_joiner(train_df)
        val_df = self._last_fm_spotify_joiner(val_df)
        test_df = self._last_fm_spotify_joiner(test_df)

        return train_df, val_df, test_df

    def _last_fm_spotify_joiner(self, df):
        artists_name_loader = StandardLoader(self.artists_name_path)
        artists_audio_loader = StandardLoader(self.artists_audio_path)
        artists_name_df = artists_name_loader.load()
        artists_audio_df = artists_audio_loader.load()

        df = df.merge(artists_name_df, on="artistID", how="left")

        df = artist_match.match_data_by_artist(
            left=df,
            right=artists_audio_df,
            left_artist_col="name",
            right_artist_col="artist_name",
            score_cutoff=self.levenshtein_score_cutoff,
        )
        return df
=======
from sklearn.pipeline import Pipeline
>>>>>>> main
