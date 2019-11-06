import logging

class Tuner(object):
    """Base class for tuners
    """

    def __init__(self, space, **kwargs):
        self.space = space
        self.params = kwargs

    def has_next(self):
        raise NotImplementedError()

    def next(self):
        raise NotImplementedError()
    
    def next_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            if not self.has_next(): break
            batch.append(self.next())
        return batch

    def update(self, inputs, result):
        pass
    
    def reset(self):
        self.best_inputs = None
        self.best_score = 0
        self.best_iter = 0

    def load_history(self, data_set):
        raise NotImplementedError()

    def tune(self, measure, n_trial, early_stopping=None, callbacks=()):
        """Begin tuning
        """
        self.reset()
        ttl = None
        n_parallel = 1
        early_stopping = early_stopping or 1e9
        i = error_ct = 0
        logging.info('begin tuning: {} {}'.format(n_trial, early_stopping))
        for i in range(n_trial):
            if not self.has_next():
                break
            inputs = self.next()
            logging.info('tuner: trial {} config: {}'.format(i, inputs))
            result = measure(inputs)
            # keep best config
            if result['error_no'] == 0:
                score = result['score']
                error_ct = 0
            else:
                score = 0
                error_ct += 1
            if score > self.best_score:
                self.best_score = score
                self.best_inputs = inputs
                self.best_iter = i
            logging.info("No: {}\t score: {:.2f}/{:.2f}".format(i+1, score, self.best_score))
            ttl = min(early_stopping + self.best_iter, n_trial) - i
            self.update(inputs, result)
            for callback in callbacks:
                callback(tuner, inputs, result)
            if i >= self.best_iter + early_stopping:
                logging.info("Early stopped : Best: %d", self.best_iter)
                break
            if error_ct > 150:
                logging.warning("Too many errors in tuning: {}".format(error_ct))