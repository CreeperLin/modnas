import logging

def tune(optim, measure, n_trial, bsize, early_stopping=None, callbacks=()):
    """Begin tuning
    """
    best_inputs = None
    best_score = 0
    best_iter = 0
    ttl = None
    n_parallel = 1
    early_stopping = early_stopping or 1e9
    i = error_ct = 0
    logging.info('tuner: start: n_trial={} early_stopping={}'.format(n_trial, early_stopping))
    for i in range(n_trial):
        if not optim.has_next():
            break
        inputs = optim.next(bsize)
        results = []
        logging.info('tuner: trial {} config: {}'.format(i, inputs))
        for inp in inputs:
            res = measure(inp)
            # keep best config
            if res['error_no'] == 0:
                score = res['score']
                error_ct = 0
            else:
                score = 0
                error_ct += 1
            if score > best_score:
                best_score = score
                best_inputs = inp
                best_iter = i
            results.append(res)
        logging.info('tuner: iter: {}\t score: {:.2f}/{:.2f}'.format(i+1, score, best_score))
        ttl = min(early_stopping + best_iter, n_trial) - i
        # optim.update(inputs, results)
        for callback in callbacks:
            callback(tuner, inputs, results)
        if i >= best_iter + early_stopping:
            logging.info('tuner: early stopped: best iter: {} score: {} config: {}'.format(best_iter, best_score, best_inputs))
            break
        if error_ct > 150:
            logging.warning('tuner: Too many errors in tuning: {}'.format(error_ct))
    logging.info('tuner: finished: best iter: {} score: {} config: {}'.format(best_iter, best_score, best_inputs))