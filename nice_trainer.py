import time
import sys
import numpy as np

INFO_STRING='{batches_done}/{batches_per_epoch} - time: {comp_time:.3f} - data: {data_time:.3f} - ETA: {eta:.0f}'


class NiceTrainer:
    def __init__(self,
                 sess,
                 bm_train,
                 feed_keys,
                 train_op,
                 bm_val=None,
                 extra_variables={},
                 printable_vars=[],
                 computed_variables={},
                 info_string=INFO_STRING):
        '''
        sess - tf session, must be initialised,
        bm_train - batch manager instance for training set
        feed_keys - are placeholders corresponding to batch returned by bm_train like (image_input, labels)
        train_op - FULL training op that will be performed every step
        loss_op - tf op that returns loss
        probs_op - some tf op that returns a value needed for extra info calculator (usually probabilities but can be anything)
        bm_val - batch manager instance for training set if you want info on validation perforance
        acc_calculator - custom function that will be called with every step that returns some accuracy metric.
                        acc_calculator(extra_vars, batch) -> float
        '''
        self.sess = sess
        self.bm_train = bm_train
        self.feed_keys = feed_keys
        self.train_op = train_op
        self.bm_val = bm_val

        self.info_string = info_string
        self.printable_vars = printable_vars
        self.measured_batches_per_sec = float("nan")

        self.fetches = [self.train_op]

        self.computed_variables = computed_variables

        self.extra_var_list = extra_variables.keys()
        self.extra_var_to_index = {e:i+len(self.fetches) for i, e in enumerate(self.extra_var_list)}

        self._extra_var_info_string = (' - ' if self.printable_vars else '') + ' - '.join('%s: {%s:.4f}' % (e, e) for e in self.printable_vars)

        self.fetches += [extra_variables[e] for e in self.extra_var_list]


    def train(self):
        ''' trains for 1 epoch'''
        smooth_burn_in = 0.9
        smooth_normal = 0.99
        smooth = None
        comp_time = None
        data_time = None
        smooth_loss = None
        custom_metric = None

        batches_per_epoch = self.bm_train.total_batches
        examples_per_epoch = self.bm_train.total_batches * self.bm_train.examples_per_batch
        batches_per_sec = float('nan')

        t_fetch = time.time()

        smoothed_printable_vars = {e:None for e in self.printable_vars}
        extra_vars = {}
        computed_vars = {}

        for batch in self.bm_train:
            t_start = time.time()
            res = self.sess.run(self.fetches, dict(zip(self.feed_keys, batch)))
            t_end = time.time()


            smooth = smooth_normal if self.bm_train.current_index > 22 else smooth_burn_in

            # now calculate all the info
            if comp_time is None:
                comp_time = t_end - t_start
            else:
                comp_time = smooth*comp_time + (1-smooth)*(t_end - t_start)

            if data_time is None:
                data_time = t_start - t_fetch
            else:
                data_time = smooth*data_time + (1-smooth)*(t_start - t_fetch)

            extra_vars = {e_var:res[self.extra_var_to_index[e_var]] for e_var in self.extra_var_list}
            extra_vars['is_training'] = True
            # now use extra_vars and batch to compute additional variables
            computed_vars = {c_var: func(extra_vars, batch) for c_var, func in self.computed_variables.items()}

            for var in self.printable_vars:
                if var in self.extra_var_list:
                    val = extra_vars[var]
                else: # must be computed
                    val = computed_vars[var]
                if smoothed_printable_vars[var] is None:
                    smoothed_printable_vars[var] = val
                else:
                    smoothed_printable_vars[var] = smoothed_printable_vars[var]*smooth + (1-smooth)*val

            batches_per_sec = 1.0 / (data_time + comp_time)
            examples_per_sec = batches_per_sec * self.bm_train.examples_per_batch

            batches_done = self.bm_train.current_index
            examples_done = self.bm_train.current_index * self.bm_train.examples_per_batch

            eta = (batches_per_epoch - batches_done) / batches_per_sec

            fraction_done = float(batches_done) / batches_per_epoch


            extra_string = self._extra_var_info_string.format(**smoothed_printable_vars)
            formatted_info_string = self.info_string.format(**locals()) + extra_string
            sys.stdout.write('\r'+ formatted_info_string)
            sys.stdout.flush()
            t_fetch = t_end

        self.measured_batches_per_sec = batches_per_sec

        print
        if self.bm_val is not None:
            self.validate()


    def validate(self):
        is_train_present = self.train_op is not None
        fetches = self.fetches[is_train_present:]   # DO NOT RUN TRAINING OP DURING VALIDATION :D
        assert self.train_op not in fetches

        extra_vars = {}
        computed_vars = {}
        averaged_extra_vars = {e: [] for e in self.printable_vars}

        for batch in self.bm_val:
            res = self.sess.run(fetches, dict(zip(self.feed_keys, batch)))

            extra_vars = {e_var: res[self.extra_var_to_index[e_var]-is_train_present] for e_var in self.extra_var_list}
            extra_vars['is_training'] = False
            # now use extra_vars and batch to compute additional variables
            computed_vars = {c_var: func(extra_vars, batch) for c_var, func in self.computed_variables.items()}

            for var in self.printable_vars:
                if var in self.extra_var_list:
                    val = extra_vars[var]
                else:  # must be computed
                    val = computed_vars[var]
                averaged_extra_vars[var].append(val)

            batches_per_epoch = self.bm_val.total_batches
            batches_done = self.bm_val.current_index

            eta = (batches_per_epoch - batches_done) / self.measured_batches_per_sec / 2  # factor of 2 because only forward pass

            formatted_info_string = 'Validation set: {batches_done}/{batches_per_epoch} - ETA: {eta:.1f}'.format(**locals())
            sys.stdout.write('\r' + formatted_info_string)
            sys.stdout.flush()

        averaged_extra_vars = {k:np.mean(v) for k,v in averaged_extra_vars.items()}
        print '\rValidation results' + self._extra_var_info_string.format(**averaged_extra_vars)




def caclulate_batch_top_n_hits(probs, labels, n):
    ''' probs is a probabilit matrix (BS, N) labels is a vector (BS,)  with every entry smaller int than N'''
    hits = 0
    for p, l in zip(probs, labels):
        hits += np.sum(p>p[l]) < n
    return hits

def accuracy_calc_op(n=1):
    def acc_op(extra_vars, batch):
        return caclulate_batch_top_n_hits(extra_vars['probs'], batch[1], n) / float(len(batch[1]))
    return acc_op