import random
import numpy as np
import os




def key_to_element(array_or_dict):
    def key_to_element_op(key):
        return array_or_dict[key]
    return key_to_element_op

def greater_than(other_value):
    def greater_than_op(value):
        return value > other_value
    return greater_than_op



def probabilistic_op(prob, op):
    ''' Selected op is applied with probability prob. Otherwise identity op is performed.'''
    def prob_op(im):
        if random.random() < prob:
            return op(im)
        else:
            return im
    return prob_op

def compose_ops(ops):
    ''' Returns a new op that applies given ops in sequence (creates pipeline)'''
    def composed_op(x):
        for op in ops:
            x = op(x)
        return x
    return composed_op

def for_each(op):
    '''Returned op takes a list as an argument and returns map(op, arg)'''
    def for_each_op(lis):
        return map(op, lis)
    return for_each_op

def parallelise_ops(ops):
    ''' Applies ops in parallel to the given input and returns a tuple of results. For example:
        ops = (op1, op2)   the new op will return (op1(inp), op2(inp))'''
    def parallelise_op(inp):
        return tuple(op(inp) for op in ops)
    return parallelise_op

def folder_name():
    def folder_name_op(path):
        return path.split(os.path.sep)[-2]
    return folder_name_op

def one_hot(classes):
    '''Converts an array to one hot representation (int32) array with shape (len(input), classes)'''
    def one_hot_op(inp):
        ret = np.zeros((len(inp), classes), dtype=np.int32)
        ret[np.arange(len(inp)), inp] = 1
        return ret
    return one_hot_op


def generic_batch_composer(type_data, type_label):
    def generic_batch_composer_op(examples):
        return parallelise_ops([
            lambda x: (np.concatenate(tuple(np.expand_dims(e[0], 0) for e in examples), 0)).astype(type_data),
            lambda x: (np.concatenate(tuple(np.expand_dims(e[1], 0) for e in examples), 0)).astype(type_label),
        ])(examples)
    return generic_batch_composer_op