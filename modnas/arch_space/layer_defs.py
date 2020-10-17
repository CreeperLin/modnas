import itertools
import torch


def get_merger(name):
    for sc in MergerBase.__subclasses__():
        if sc.__name__ == name:
            return sc
    raise NotImplementedError('unsupported merger: {}'.format(name))


class MergerBase():
    def __init__(self):
        pass

    def chn_out(self, chn_states):
        pass

    def merge(self, states):
        pass

    def merge_range(self, num_states):
        pass


class ConcatMerger(MergerBase):
    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        return sum(chn_states[self.start:])

    def merge(self, states):
        return torch.cat(states[self.start:], dim=1)

    def merge_range(self, num_states):
        return range(self.start, num_states)


class AvgMerger(MergerBase):
    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        return chn_states[-1]

    def merge(self, states):
        return sum(states[self.start:]) / len(states)

    def merge_range(self, num_states):
        return range(self.start, num_states)


class SumMerger(MergerBase):
    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        return chn_states[-1]

    def merge(self, states):
        return sum(states[self.start:])

    def merge_range(self, num_states):
        return range(self.start, num_states)


class LastMerger(MergerBase):
    def __init__(self, start=0):
        del start
        super().__init__()

    @staticmethod
    def chn_out(chn_states):
        return chn_states[-1]

    @staticmethod
    def merge(states):
        return states[-1]

    @staticmethod
    def merge_range(num_states):
        return (num_states - 1, )


def get_enumerator(name):
    for sc in EnumeratorBase.__subclasses__():
        if sc.__name__ == name:
            return sc
    raise NotImplementedError('unsupported enumerator: {}'.format(name))


class EnumeratorBase():
    def __init__(self):
        pass

    def enum(self, n_states, n_inputs):
        pass

    def len_enum(self, n_states, n_inputs):
        pass


class CombinationEnumerator(EnumeratorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def enum(n_states, n_inputs):
        return itertools.combinations(range(n_states), n_inputs)

    @staticmethod
    def len_enum(n_states, n_inputs):
        return len(list(itertools.combinations(range(n_states), n_inputs)))


class LastNEnumerator(EnumeratorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def enum(n_states, n_inputs):
        yield [n_states - i - 1 for i in range(n_inputs)]

    @staticmethod
    def len_enum(n_states, n_inputs):
        return 1


class FirstNEnumerator(EnumeratorBase):
    def __init__(self):
        super().__init__()

    @staticmethod
    def enum(n_states, n_inputs):
        yield [i for i in range(n_inputs)]

    @staticmethod
    def len_enum(n_states, n_inputs):
        return 1


class TreeEnumerator(EnumeratorBase):
    def __init__(self, width=2):
        super().__init__()
        self.width = width

    def enum(self, n_states, n_inputs):
        yield [(n_states - 1 + i) // self.width for i in range(n_inputs)]

    def len_enum(self, n_states, n_inputs):
        return 1


class N2OneEnumerator(EnumeratorBase):
    def __init__(self):
        super().__init__()

    def enum(self, n_states, n_inputs):
        yield [n_states - i - 1 for i in range(n_states)]

    def len_enum(self, n_states, n_inputs):
        return 1


def get_allocator(name):
    for sc in AllocatorBase.__subclasses__():
        if sc.__name__ == name:
            return sc
    raise NotImplementedError('unsupported allocator: {}'.format(name))


class AllocatorBase():
    def __init__(self):
        pass

    def alloc(self, states, sidx, cur_state):
        pass

    def chn_in(self, chn_states, sidx, cur_state):
        pass


class EvenSplitAllocator(AllocatorBase):
    def __init__(self, n_inputs, n_states):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.tot_states = n_inputs + n_states
        self.slice_map = {}

    def alloc(self, states, sidx, cur_state):
        ret = []
        for s, si in zip(states, sidx):
            s_slice = self.slice_map[(si, cur_state)]
            s_in = s[:, s_slice]
            ret.append(s_in)
        return ret

    def chn_in(self, chn_states, sidx, cur_state):
        chn_list = []
        for (chn_s, si) in zip(chn_states, sidx):
            etot = min(self.n_states, self.tot_states - si - 1)
            eidx = cur_state - max(self.n_inputs, si + 1)
            c_in = chn_s - (chn_s // etot) * eidx if eidx == etot - 1 else chn_s // etot
            chn = chn_s // etot
            end = chn_s if eidx == etot - 1 else chn * (eidx + 1)
            s_slice = slice(chn * eidx, end)
            self.slice_map[(si, cur_state)] = s_slice
            chn_list.append(c_in)
        return chn_list


class FracSplitAllocator(AllocatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def alloc(self, states, sidx, cur_state):
        return states

    def chn_in(self, chn_states, sidx, cur_state):
        return chn_states


class ReplicateAllocator(AllocatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def alloc(self, states, sidx, cur_state):
        return states

    def chn_in(self, chn_states, sidx, cur_state):
        return chn_states
