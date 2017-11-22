import argparse
import pickle

import tensorflow as tf


def files_to_tfrecord_fixedlen(*files, out_path, seq_len):
    """
    Process a number of text files into TFRecords data file. All files are conjoined into one big string. For 
    simplicity, we split this string into equal-length sequences of seq_len-1 characters each. Furthermore, a special
    "beginning-of-sequence" character is prepended to each sequence, and the characters are mapped to integer indices
    representing one-hot vectors.
    """
    full_text = "\n".join(open(file).read() for file in files)
    # we create a mapping from characters to integers, including a special "beginning of sequence" character
    chars = set(full_text)
    ch_to_ind = dict(zip(chars, range(1, len(chars)+1)))
    ch_to_ind["<S>"] = 0

    seqs = text_to_seqs(full_text, seq_len, ch_to_ind)
    print("Split input into {} sequences...".format(len(seqs)))

    with tf.python_io.TFRecordWriter(out_path + ".tfrecords") as writer:
        for ind, seq in enumerate(seqs):
            tfex = tf.train.Example(features=tf.train.Features(feature={
                "seq": tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
            }))
            writer.write(tfex.SerializeToString())
            if (ind + 1) % 100 == 0:
                print("Serialized {} sequences...".format(ind+1))
    pickle.dump(ch_to_ind, open(out_path + "_vocab", mode="wb"))


def text_to_seqs(text, seq_len, mapping):
    """
    Convert a string to a list of lists of equal length. Each character is mapped to its indexed as given by the mapping 
    parameter.
    Right now this will actually use sequences *one character shorter* than requested, but prepend a "beginning of
    sequence" character.
    """
    use_bos = True
    if use_bos:
        seq_len -= 1
    seqs = [[mapping["<S>"]] + chs_to_inds(text[ind:(ind+seq_len)], mapping) for ind in range(0, len(text), seq_len)]
    # we throw away the last "leftover" sequence if it's shorter
    return seqs[:-1] if len(seqs[-1]) != len(seqs[0]) else seqs


def chs_to_inds(char_list, mapping):
    """Helper to convert a list of characters (or just a string) to a list of corresponding indices."""
    return [mapping[ch] for ch in char_list]


def parse_seq(example_proto, seq_len):
    """Needed to read the stored .tfrecords data -- import this in your training script."""
    features = {"seq": tf.FixedLenFeature((seq_len,), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["seq"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files", help="File paths to use as input, separated by commas. E.g. "
                                           "'file1.txt,file2.txt'.")
    parser.add_argument("out_path", help="Path to store the data to. Do *not* specify the file extension, as this "
                                         "script stores both a .tfrecords file as well as a vocabulary file.")
    parser.add_argument("-l", "--seqlen", type=int, default=200, help="How many characters per sequence. Default: 200.")
    args = parser.parse_args()
    file_list = args.data_files.split(",")
    files_to_tfrecord_fixedlen(*file_list, out_path=args.out_path, seq_len=args.seqlen)
