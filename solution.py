import tensorflow as tf
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

os.environ['TF_DETERMINISTIC_OPS'] = '1'

# The batch size for training
batch_size = 256
# The sequence length for training
seq_length = 4
# The number of units in each LSTM layer
rnn_units = 192
# The number of LSTM layers
num_layers = 1
# The size of the embedding layer
embedding_size = 1024
# The initial learning rate for optimizer
start_learning_rate = 0.005
# The final learning rate for optimizer
end_learning_rate = 0.001
# The mode for the program, "compress", "decompress", "both"
mode = 'both'

path_to_file = "data_from_solution/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


def build_model(vocab_size: int) -> tf.keras.Model:
    """Builds the model architecture with GRU instead of LSTM.

    Args:
      vocab_size: Int, size of the vocabulary.
    """
    inputs = [
        tf.keras.Input(shape=[seq_length], batch_size=batch_size)
    ]
    for _ in range(num_layers):
        inputs.append(tf.keras.Input(shape=(rnn_units,)))
        inputs.append(tf.keras.Input(shape=(rnn_units,)))

    embedding = tf.keras.layers.Embedding(
        vocab_size, embedding_size, name="embedding"
    )(inputs[0])

    skip_connections = []
    outputs = []

    h0_in, c0_in = inputs[1], inputs[2]
    init0 = tf.keras.layers.Lambda(
        lambda x: x[0] + 0.0 * x[1],
        name="gru0_init"
    )([h0_in, c0_in])

    gru0 = tf.keras.layers.GRU(
        rnn_units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer="glorot_uniform",
        name="gru_0",
    )
    predictions, state_h = gru0(embedding, initial_state=init0)

    skip_connections.append(predictions)
    outputs.append(state_h)
    outputs.append(state_h)

    for layer_idx in range(num_layers - 1):
        h_in = inputs[layer_idx * 2 + 3]
        c_in = inputs[layer_idx * 2 + 4]
        init = tf.keras.layers.Lambda(
            lambda x: x[0] + 0.0 * x[1],
            name=f"gru{layer_idx+1}_init"
        )([h_in, c_in])

        layer_input = tf.keras.layers.Concatenate(name=f"concat_{layer_idx+1}")(
            [embedding, skip_connections[-1]]
        )

        gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            name=f"gru_{layer_idx+1}",
        )
        predictions, state_h = gru(layer_input, initial_state=init)

        skip_connections.append(predictions)
        outputs.append(state_h)
        outputs.append(state_h)

    last_timestep = []
    for i in range(num_layers):
        last_timestep.append(
            tf.keras.layers.Lambda(
                lambda x: x[:, seq_length - 1, :],
                name=f"last_timestep_{i}"
            )(skip_connections[i])
        )

    if num_layers == 1:
        layer_input = last_timestep[0]
    else:
        layer_input = tf.keras.layers.Concatenate(
            name="final_concat"
        )(last_timestep)

    dense = tf.keras.layers.Dense(vocab_size, name="dense_logits")(layer_input)
    output = tf.keras.layers.Activation(
        "softmax", dtype="float32", name="predictions"
    )(dense)

    outputs.insert(0, output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_compressor")
    return model


def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress,
          data, states):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the *first* batch.
        seq_input: Tensor, containing the last seq_length inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.
        states: List containing state information for the layers of the model.

    Returns:
        seq_input: Tensor, containing the last seq_length inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    loss = cross_entropy = denom = 0.0
    split = math.ceil(length / batch_size)
    with tf.GradientTape() as tape:
        inputs = states.pop(0)
        inputs.insert(0, seq_input)
        outputs = model(inputs)
        predictions = outputs.pop(0)
        states.append(outputs)

        p = predictions.numpy()
        batch_indices = pos + 1 + np.arange(batch_size) * split
        freq_all = np.cumsum(p * 10000000.0 + 1.0, axis=1)

        symbols = []
        mask_vals = []
        for i, index in enumerate(batch_indices):
            freq = freq_all[i]
            symbol = get_symbol(index, length, freq, coder, compress, data)
            symbols.append(symbol)
            if index < length:
                prob = p[i, symbol]
                if prob <= 0.0:
                    prob = 0.000001
                cross_entropy += math.log2(prob)
                denom += 1.0
                mask_vals.append(1.0)
            else:
                mask_vals.append(0.0)

        symbols_arr = np.array(symbols, dtype=np.int32)
        mask_tensor = tf.convert_to_tensor(mask_vals, dtype=predictions.dtype)
        input_one_hot = tf.one_hot(symbols_arr, vocab_size)

        loss = tf.keras.losses.categorical_crossentropy(
            input_one_hot, predictions, from_logits=False) * tf.expand_dims(
                mask_tensor, 1)

        seq_input = tf.concat(
            [seq_input[:, 1:], tf.expand_dims(symbols_arr, 1)],
            axis=1
        )

    gradients = tape.gradient(loss, model.trainable_variables)
    capped_grads = [tf.clip_by_norm(grad, 4.0) for grad in gradients]
    optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))
    return seq_input, cross_entropy, denom


def reset_seed():
    """Initializes various random seeds to help with determinism."""
    SEED = 1234
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def process(compress, length, vocab_size, coder, data):
    """This runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()
    reset_seed()
    model = build_model(vocab_size=vocab_size)
    model.summary()

    split = math.ceil(length / batch_size)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        split,
        end_learning_rate,
        power=1.0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn, beta_1=0, beta_2=0.9999, epsilon=1e-5)

    freq = np.cumsum(np.full(vocab_size, 1.0 / vocab_size) * 10000000.0 + 1.0)
    batch_offsets = np.arange(batch_size) * split
    symbols = [
        get_symbol(index, length, freq, coder, compress, data)
        for index in batch_offsets
    ]
    symbols_arr = np.array(symbols, dtype=np.int32)
    seq_input = tf.tile(tf.expand_dims(symbols_arr, 1), [1, seq_length])

    pos = 0
    cross_entropy = 0.0
    denom = 0.0
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'

    states = [
        [tf.zeros([batch_size, rnn_units])] * (num_layers * 2)
        for _ in range(seq_length)
    ]

    while pos < split:
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model,
                                 optimizer, compress, data, states)
        cross_entropy += ce
        denom += d
        pos += 1
        if pos % 5 == 0:
            percentage = 100.0 * pos / split
            if percentage >= 100.0:
                continue
            print(template.format(percentage, -cross_entropy / max(denom, 1.0), time.time() - start))
    if compress:
        coder.finish()
    print(template.format(100.0, -cross_entropy / max(length, 1.0), time.time() - start))


def compession():
    # int_list will contain the characters of the file.
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}
    for c in text:
        int_list.append(char2idx[c])

    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size/8) * 8
    print('Length of file: {} symbols'.format(len(int_list)))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        # Write 256 bits to the compressed file header to keep track of the vocabulary.
        for i in range(256):
            bitout.write(1 if i in char2idx else 0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)


def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        # Read the original file size from the header.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        # Create a list to store the file characters.
        output = [0] * length
        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        # Round up to a multiple of 8 to improve performance.
        vocab_size = math.ceil(vocab_size/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        # The decompressed data is stored in the "output" list. We can now write the
        # data to file (based on the type of preprocessing used).

        idx2char = np.array(vocab, dtype=np.uint8)
        out.write(idx2char[output].tobytes())


def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file)/os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()
