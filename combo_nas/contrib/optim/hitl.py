from collections import OrderedDict
import combo_nas.optim as optim
from combo_nas.optim.base import OptimBase
from combo_nas.core.param_space import ParamNumeric

@optim.register_as('HITL')
class HITLParamOptim(OptimBase):
    '''
        Human-in-the-loop Optimizer
        for debugging use
    '''
    def has_next(self):
        return True

    def parse_input(self, param, inp):
        if isinstance(param, ParamNumeric):
            return float(inp)
        try:
            return int(inp)
        except ValueError:
            return inp

    def check_value(self, param, value):
        if value is None:
            return False
        return param.is_valid(value)

    def _next(self):
        ret = OrderedDict()
        for name, param in self.space.named_params():
            prompt = '{}\nvalue: '.format(str(param))
            while True:
                inp = input(prompt)
                value = self.parse_input(param, inp)
                if self.check_value(param, value):
                    break
                print('invalid input')
            ret[name] = value
        return ret

    def step(self, estim):
        inputs, results = estim.get_last_results()
        self.logger.info('update:\ninputs: {}\nresults: {}'.format(inputs, results))
