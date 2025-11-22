from interfaces.base_interaction_matrix_builder import BaseMatrixBuilder


class InteractionMatrixBuilder(BaseMatrixBuilder):
    def build(self, interactions_df, users, items):
        """
        Returns:
          interactions       -> sparse matrix [n_users x n_items]
          weights (optional) -> sparse matrix with interaction strength
        """
        ...
