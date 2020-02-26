from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

def my_pad(sequence, maxlen=30, pad_id=0, padding='post', truncating='post'):
    
    if not len(sequence):
        return []
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    elif truncating == 'post':
        trunc = sequence[:maxlen]
    
    x = [pad_id] * maxlen
    
    if padding == 'post':
        x[:len(trunc)] = trunc
    elif padding == 'pre':
        x[-len(trunc):] = trunc
    
    return x

def keras_pad_fn(token_ids_batch, maxlen, pad_id=0, padding='post', truncating='post'):
    padded_token_ids_batch = pad_sequences(token_ids_batch,
                                            value=pad_id,  # vocab.transform_token2idx(PAD),
                                            padding=padding,
                                            truncating=truncating,
                                            maxlen=maxlen)
    return padded_token_ids_batch

# pad_sequences_fn in keras.preprocessing.sequence.pad_sequences
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


if __name__ == '__main__':
    sequences = [[2, 4, 62], [2,35,12,24,2]]
    pad_res = pad_sequences(sequences, maxlen=10, dtype='int32', padding='pre', truncating='post', value=0.)
    keras_pad_res = keras_pad_fn(sequences, maxlen=10, pad_id=0, padding='post', truncating='post')
    print(pad_res)
    print(keras_pad_res)