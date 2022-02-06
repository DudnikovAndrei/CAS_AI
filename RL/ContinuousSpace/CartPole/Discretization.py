
import numpy as np

class Discretization():
    def create_uniform_grid(self, low, high, bins):
        """Define a uniformly-spaced grid that can be used to discretize a space.

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of the continuous space.
        high : array_like
            Upper bounds for each dimension of the continuous space.
        bins : tuple
            Number of bins along each corresponding dimension.

        Returns
        -------
        grid : list of array_like
            A list of arrays containing split points for each dimension.
        """
        grids = []
        for low, high, bin in zip(low, high, bins):
            grids.append(np.linspace(low, high, bin+1)[1:-1])
        return grids

    def discretize(self, sample, grid):
        """Discretize a sample as per given grid.

        Parameters
        ----------
        sample : array_like
            A single sample from the (original) continuous space.
        grid : list of array_like
            A list of arrays containing split points for each dimension.

        Returns
        -------
        discretized_sample : array_like
            A sequence of integers with the same number of dimensions as sample.
        """
        # TODO: Implement this
        return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension

    def transform_cart_simples_to_array(self, samples):
        cart_samples = []
        for sample in samples:
            cart_samples.append(list(sample[:2]))
        return np.asarray(cart_samples)

    def transform_pole_simples_to_array(self, samples):
        pole_samples = []
        for sample in samples:
            pole_samples.append(list(sample[2:]))
        return np.asarray(pole_samples)

    def transform_cart_discretized_to_array(self, discretized_samples):
        cart_discretized = []
        for sample in discretized_samples:
            cart_discretized.append(list(sample[:2]))
        return np.asarray(cart_discretized)

    def transform_pole_discretized_to_array(self, discretized_samples):
        pole_discretized = []
        for sample in discretized_samples:
            pole_discretized.append(list(sample[2:]))
        return np.asarray(pole_discretized)