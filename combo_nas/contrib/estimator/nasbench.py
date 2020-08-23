import torch.nn as nn
from combo_nas.core.param_space import ArchParamCategorical
from combo_nas.estimator.predefined.regression_estimator import RegressionEstimator, ArchPredictor
import combo_nas.arch_space
import combo_nas.estimator
try:
    from nasbench import api
except ImportError:
    api = None

INPUT = 'input'
OUTPUT = 'output'

CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


@combo_nas.arch_space.register_as('NASBench')
class NASBenchNet(nn.Module):
    def __init__(self, n_nodes=7):
        super().__init__()
        matrix = []
        ops = []
        n_states = n_nodes - 2
        n_edges = n_nodes * (n_nodes - 1) // 2
        for _ in range(n_edges):
            matrix.append(ArchParamCategorical(['0', '1']))
        for _ in range(n_states):
            ops.append(ArchParamCategorical([CONV1X1, CONV3X3, MAXPOOL3X3]))
        self.matrix_params = matrix
        self.ops_params = ops

    def to_arch_desc(self):
        matrix = [p.value() for p in self.matrix_params]
        ops = [p.value() for p in self.ops_params]
        return (matrix, ops)


class NASBenchPredictor(ArchPredictor):
    def __init__(self, record_path):
        super().__init__()
        if api is None:
            raise RuntimeError('nasbench api is not installed')
        self.nasbench = api.NASBench(record_path)
        self.max_nodes = 7

    def fit(self, ):
        pass

    def predict(self, arch_desc):
        max_nodes = self.max_nodes
        matrix = [[0] * max_nodes for i in range(max_nodes)]
        g_matrix = arch_desc[0]
        k = 0
        for i in range(max_nodes):
            for j in range(i + 1, max_nodes):
                matrix[i][j] = int(g_matrix[k])
                k += 1
        ops = [INPUT] + arch_desc[1] + [OUTPUT]
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        try:
            data = self.nasbench.query(model_spec)
            val_acc = data['test_accuracy']
        except:
            val_acc = 0
        return val_acc


@combo_nas.estimator.register_as('NASBench')
class NASBenchEstimator(RegressionEstimator):
    def run(self, optim):
        config = self.config
        self.logger.info('loading NASBench data')
        self.predictor = NASBenchPredictor(config.record_path)
        return super().run(optim)
