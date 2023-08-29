from typing import List

import numpy

class BIPOP(Restart):
    def __init__(self, dimension: int, lamb: float, mu: float, budget: int) -> None: ...
    def large(self) -> bool: ...
    @property
    def budget(self) -> int: ...
    @property
    def budget_large(self) -> int: ...
    @property
    def budget_small(self) -> int: ...
    @property
    def lambda_init(self) -> int: ...
    @property
    def lambda_large(self) -> int: ...
    @property
    def lambda_small(self) -> int: ...
    @property
    def mu_factor(self) -> float: ...
    @property
    def used_budget(self) -> int: ...

class IPOP(Restart):
    ipop_factor: float
    def __init__(self, dimension: int, lamb: float) -> None: ...

class NoRestart(Strategy):
    def __init__(self) -> None: ...
    def evaluate(self, parameters) -> None: ...

class Restart(Strategy):
    def __init__(self, dimension: int, lamb: float) -> None: ...
    def evaluate(self, parameters) -> None: ...
    def restart(self, parameters) -> None: ...
    def setup(self, dimension: float, lamb: float, t: int) -> None: ...
    def termination_criteria(self, parameters) -> bool: ...
    @property
    def best_fitnesses(self) -> List[float]: ...
    @property
    def flat_fitness_index(self) -> int: ...
    @property
    def flat_fitnesses(self) -> numpy.ndarray[numpy.int32[m,1]]: ...
    @property
    def last_restart(self) -> int: ...
    @property
    def max_iter(self) -> int: ...
    @property
    def median_fitnesses(self) -> List[float]: ...
    @property
    def n_bin(self) -> int: ...
    @property
    def n_stagnation(self) -> int: ...

class Strategy:
    ...