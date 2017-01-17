import os
import urllib2
import StringIO
import tarfile
from batch_manager import BatchManager
from bs4 import BeautifulSoup
from unidecode import unidecode
import numpy as np
from text_pipeline import *
from collections import Counter


random.seed(33)
GEN_SPAM_DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'GenSpam')
GEN_SPAM_DATA_URL = 'http://www.cl.cam.ac.uk/Research/NL/nl-download/GenSpam.tar.gz'
VOCAB_SIZE = 5000
VOCAB = None
R_VOCAB = None
LOWER_VOCAB = None
UNKNOWN_WORD = '#?#'
PADDING_CHAR = '~'

def remove_confusing_tags():
    def remove_confusing_tags_op(msg):
        parsed = BeautifulSoup(msg.decode('utf-8', 'ignore'), 'lxml')
        try:
            subject = parsed.find('subject')
            if subject is None:
                subject = ''
            else:
                subject = subject.text.strip()
            body = parsed.find('message_body')
            body = body.text.strip() if body is not None else ''
        except:
            print parsed
            raise
        txt = '#SUBJ %s #SUBJ\n%s #END' % (subject, body)
        for e in txt:
            if not ord(e) < 255:
                txt = unidecode(txt)
                break
        return txt

    return remove_confusing_tags_op


def maybe_download_and_extract():
    if os.path.exists(GEN_SPAM_DATA_FOLDER):
        return
    print 'Could not find GenSpam dataset. Please wait, downloading...'
    os.mkdir(GEN_SPAM_DATA_FOLDER)
    data = StringIO.StringIO(urllib2.urlopen(GEN_SPAM_DATA_URL).read())
    tar = tarfile.open(fileobj=data, mode="r:gz")
    tar.extractall(GEN_SPAM_DATA_FOLDER)
    tar.close()
    print 'GenSpam downloaded and ready to use.'


def maybe_build_vocab():
    global VOCAB, R_VOCAB, LOWER_VOCAB
    if VOCAB is not None:
        return
    maybe_download_and_extract()

    gen, spam = load_messages('train')

    words = ' '.join(gen+spam).split()
    counted = Counter(words)
    vocab = sorted(set(words), key=lambda x: counted[x], reverse=True)[:VOCAB_SIZE-1] + [UNKNOWN_WORD]

    R_VOCAB = dict(enumerate(vocab))
    VOCAB = {v:k for k, v in R_VOCAB.items()}
    assert '~' in VOCAB # padding must be present!
    LOWER_VOCAB = {}
    for k, v in VOCAB.items():
        l = k.lower()
        if k==l:
            LOWER_VOCAB[l] = v
        else:
            if l in VOCAB or l in LOWER_VOCAB:
                continue # this will be added later
            else:
                LOWER_VOCAB[l] = v

def to_word_code(message):
    maybe_build_vocab()
    m = message.split()

    for i, w in enumerate(m):
        if w in VOCAB:
            m[i] = VOCAB[w]
        elif w.lower() in LOWER_VOCAB:
            m[i] = LOWER_VOCAB[w.lower()]
        else:
            m[i] = VOCAB[UNKNOWN_WORD]
    return m

def from_word_code(code):
    maybe_build_vocab()
    return ' '.join(map(lambda x: R_VOCAB[x], code))

def load_messages(*types):
    def load_path(path):
        text = open(path).read()
        for e in text:
            assert ord(e) < 256 # we only accept ASCII
        split = map(lambda x: x + '</MESSAGE>', text.split('</MESSAGE>')[:-1])
        return split
    return sum([load_path(os.path.join(GEN_SPAM_DATA_FOLDER, '%s_GEN.ems' % typ)) for typ in types], []), sum([load_path(os.path.join(GEN_SPAM_DATA_FOLDER, '%s_SPAM.ems' % typ)) for typ in types], [])


def get_train_bm(message_length, minimal_actual_text_length, batch_size, end_padding='~', shuffle_examples=True, num_workers=1, word_code=False):
    maybe_download_and_extract()

    gen_adapt, spam_adapt = load_messages('adapt')
    gen_train, spam_train = load_messages('train')
    random.shuffle(spam_train)
    gen, spam = gen_adapt + gen_train, spam_adapt + spam_train


    data_len = min(len(gen), len(spam))
    gen, spam = gen[:data_len], spam[:data_len]
    messages = tuple(gen) + tuple(spam)
    indices = tuple(range(len(messages)))

    if not word_code: # char code
        # pipeline is key -> message -> remove useless tags -> randomly cropped message of fixed length -> char_codes
        MESSAGE_TRAIN_PIPELINE = compose_ops([
            key_to_element(messages),
            remove_confusing_tags(),
            random_crop(message_length, minimal_actual_text_length, front_padding=None, end_padding=end_padding),
            for_each(ord),
        ])
    else:
        maybe_build_vocab()
        # pipeline is key -> message -> word_code -> randomly cropped message of fixed length
        MESSAGE_TRAIN_PIPELINE = compose_ops([
            key_to_element(messages),
            to_word_code,
            random_crop(message_length, minimal_actual_text_length, front_padding=None, end_padding=[VOCAB[PADDING_CHAR]]),
        ])
    LABEL_TRAIN_PIPELINE = compose_ops([
        greater_than(len(gen)-1),
        key_to_element({False: 0, True: 1})
    ])
    SPAM_GEN_EXAMPLE_GETTER = for_each(parallelise_ops([MESSAGE_TRAIN_PIPELINE, LABEL_TRAIN_PIPELINE]))

    return BatchManager(SPAM_GEN_EXAMPLE_GETTER,
                        indices,
                        generic_batch_composer(np.int32, np.int32),
                        batch_size,
                        shuffle_examples=shuffle_examples,
                        num_workers=num_workers)



def get_val_bm(message_length, minimal_actual_text_length, batch_size, end_padding='~', shuffle_examples=True, num_workers=1, word_code=False):
    maybe_download_and_extract()

    gen, spam = load_messages('test')

    messages = tuple(gen) + tuple(spam)
    indices = tuple(range(len(messages)))

    if not word_code:
        # pipeline is key -> message -> randomly cropped message of fixed length -> char_codes
        MESSAGE_TRAIN_PIPELINE = compose_ops([
            key_to_element(messages),
            remove_confusing_tags(),
            #random_crop(message_length, minimal_actual_text_length, front_padding=None, end_padding=end_padding),
            beg_crop_with_padding(message_length),
            for_each(ord),
        ])
    else:
        maybe_build_vocab()
        # pipeline is key -> message -> word_code -> randomly cropped message of fixed length
        MESSAGE_TRAIN_PIPELINE = compose_ops([
            key_to_element(messages),
            to_word_code,
            beg_crop_with_padding(message_length, [VOCAB[PADDING_CHAR]])
            #random_crop(message_length, minimal_actual_text_length, front_padding=None, end_padding=[VOCAB[PADDING_CHAR]]),
        ])
    LABEL_TRAIN_PIPELINE = compose_ops([
        greater_than(len(gen) - 1),
        key_to_element({False: 0, True: 1})
    ])
    SPAM_GEN_EXAMPLE_GETTER = for_each(parallelise_ops([MESSAGE_TRAIN_PIPELINE, LABEL_TRAIN_PIPELINE]))

    return BatchManager(SPAM_GEN_EXAMPLE_GETTER,
                        indices,
                        generic_batch_composer(np.int32, np.int32),
                        batch_size,
                        shuffle_examples=shuffle_examples,
                        num_workers=num_workers)

def get_r_vocab():
    maybe_build_vocab()
    return R_VOCAB


if __name__=='__main__':
    from collections import Counter
    i =0
    a = get_val_bm(256, 256, 16, word_code=1)
    f = []
    g = []
    for e in a:
        for x in xrange(len(e[0])):
            f.append(' '.join(map(lambda x: R_VOCAB[x], e[0][x])) + str(e[1][x]))
            g.append(e[1][x])
    print a.total_batches
    c = Counter(f)
    print len(dict(c))
    m = max(c.values())
    print 'Max dup', m
    for k, v in c.items():
        if v==m:
            print k
            print
    for e in xrange(10):
        print
        print f[e]
    print g.count(1) / float(len(g))

