import itertools
from .regression_estimator import RegressionEstimator, ArchPredictor
from ...core.param_space import ArchParamSpace, ArchParamDiscrete, ArchParamContinuous
try:
    from nasbench import api
except:
    api = None

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

class NASBenchNet():
    def __init__(self, n_nodes=7):
        matrix = []
        ops = []
        n_states = n_nodes - 2
        n_edges = n_nodes * (n_nodes-1) // 2
        for i in range(n_edges):
            matrix.append(ArchParamDiscrete([0, 1]))
        for i in range(n_states):
            ops.append(ArchParamDiscrete([CONV1X1, CONV3X3, MAXPOOL3X3]))
        self.matrix_params = matrix
        self.ops_params = ops
    
    def to_genotype(self):
        matrix = [p.value() for p in self.matrix_params]
        ops = [p.value() for p in self.ops_params]
        return (
            matrix,
            ops
        )


class NASBenchPredictor(ArchPredictor):
    def __init__(self, record_path):
        super().__init__()
        if api is None:
            raise RuntimeError('nasbench api is not installed')
        self.nasbench = api.NASBench(record_path)
        self.max_nodes = 7
    
    def fit(self, ):
        pass
    
    def predict(self, genotype):
        max_nodes = self.max_nodes
        matrix = [[0]*max_nodes for i in range(max_nodes)]
        g_matrix = genotype[0]
        k = 0
        for i in range(max_nodes):
            for j in range(i+1, max_nodes):
                matrix[i][j] = g_matrix[k]
                k+=1
        ops = [INPUT] + genotype[1] + [OUTPUT]
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        try:
            data = self.nasbench.query(model_spec)
            val_acc = data['test_accuracy']
        except:
            val_acc = 0
        return val_acc


class NASBenchEstimator(RegressionEstimator):

    def search(self, arch_optim):
        config = self.config
        self.logger.info('generating NASBench param space')
        self.model = NASBenchNet()
        self.logger.info('loading NASBench data')
        self.predictor = NASBenchPredictor(config.record_path)
        return super().search(arch_optim)