import random
from general_data_pipeline import *

def beg_crop_with_padding(length, padding='~'):
    def beg_crop_with_padding_op(txt):
        c = txt[:length]
        if len(c) < length:
            c += (length-len(c))*padding
        return c
    return beg_crop_with_padding_op


def random_crop(length, min_actual_length=None, front_padding=None, end_padding=None):
    assert min_actual_length is None or min_actual_length <= length
    assert front_padding is None or len(front_padding)==1
    assert end_padding is None or len(end_padding)==1
    front_padding = front_padding if front_padding else ''
    end_padding = end_padding if end_padding else ''
    min_actual_text_length = min_actual_length if min_actual_length is not None else 0
    def random_crop_op(text):
        if len(text) < length and not front_padding and not end_padding:
            raise ValueError('Some text example has length (%d) smaller than required length (%d) and no padding was provided' % (len(text), length))

        requested_min_text_length = min(len(text), min_actual_text_length)
        start_index = random.randrange(requested_min_text_length-length, len(text)+1-requested_min_text_length)
        if not front_padding:
            start_index = max(0, start_index)
        if not end_padding:
            start_index = min(start_index, len(text)-length)

        actual_text =  text[max(0, start_index): start_index+length]
        assert len(actual_text) >= requested_min_text_length

        front_insert = (-start_index)*front_padding if start_index < 0 else type(text)()
        end_insert =  (start_index + length - len(text))*end_padding if start_index + length > len(text) else type(text)()

        final_crop = front_insert + actual_text + end_insert
        assert len(final_crop) == length

        return final_crop
    return random_crop_op


