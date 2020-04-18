import numpy as np
import random
from ..base import CategoricalSpaceOptim

class GeneticOptim(CategoricalSpaceOptim):
    def __init__(self, space, pop_size, logger=None):
        super().__init__(space, logger)
        self.pop_size = pop_size
        self.operators = []
        self.metrics = []
        self.population = self._initialize()

    def _initialize(self):
        raise NotImplementedError

    def _mating(self, pop):
        cur_pop = pop
        for op in self.operators:
            cur_pop = op(cur_pop)
        return cur_pop

    def _next(self):
        return self.population[len(self.metrics)]

    def add_operator(self, operator):
        self.operators.append(operator)

    def to_metrics(self, res):
        if isinstance(res, dict):
            return list(res.values())[0]
        if isinstance(res, (tuple, list)):
            return res[0]
        return res

    def step(self, estim):
        _, results = estim.get_last_results()
        results = [self.to_metrics(res) for res in results]
        self.metrics.extend(results)
        if len(self.metrics) >= len(self.population):
            self.population = self._mating(self.population)
            self.metrics = []


class EvolutionOptim(GeneticOptim):
    def __init__(self, space, pop_size=100, n_parents=2, n_offsprings=1,
                 n_select=10, n_survival=None, n_crossover=None,
                 mutation_prob=0.1, logger=None):
        super().__init__(space, pop_size, logger)
        self.add_operator(self._survival)
        self.add_operator(self._selection)
        self.add_operator(self._crossover)
        self.add_operator(self._mutation)
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.n_select = n_select
        self.n_survival = pop_size if n_survival is None else n_survival
        self.n_crossover = pop_size if n_crossover is None else n_crossover
        self.mutation_prob = mutation_prob

    def _initialize(self):
        return [self.get_random_params() for _ in range(self.pop_size)]

    def _survival(self, pop):
        n_survival = self.n_survival
        if n_survival >= len(pop):
            return pop
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_survival)[-n_survival:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _selection(self, pop):
        n_select = self.n_select
        metrics = np.array(self.metrics)
        idx = np.argpartition(metrics, -n_select)[-n_select:]
        self.metrics = [metrics[i] for i in idx]
        return [pop[i] for i in idx]

    def _crossover(self, pop):
        next_pop = []
        while len(next_pop) < self.n_crossover:
            parents = [random.choice(pop) for _ in range(self.n_parents)]
            for _ in range(self.n_offsprings):
                n_gene = parents[0].copy()
                for name in parents[0]:
                    values = [p[name] for p in parents]
                    n_gene[name] = random.choice(values)
                next_pop.append(n_gene)
        return next_pop

    def _mutation(self, pop):
        next_pop = []
        for gene in pop:
            m_gene = gene.copy()
            for name, value in gene.items():
                p = self.space.get_param(name)
                if random.random() < self.mutation_prob:
                    nidx = idx = p.get_index(value)
                    while nidx == idx:
                        nidx = random.randint(0, len(p) - 1)
                    m_gene[name] = p.get_value(nidx)
            next_pop.append(m_gene)
        return next_pop


class RegularizedEvolutionOptim(EvolutionOptim):
    def _survival(self, pop):
        n_survival = self.n_survival
        s_idx = len(pop) - n_survival
        if s_idx < 0:
            return pop
        self.metrics = self.metrics[s_idx:]
        return pop[s_idx:]
