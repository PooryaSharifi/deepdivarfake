import tensorflow as tf
import numpy as np
import os.path
import time
hidden_size = 64
unify_dim = 64
vocab_size = 8192


def generator():
    input_layer = tf.keras.layers.Input((64,))
    embedding_layer = tf.keras.layers.Embedding(vocab_size, unify_dim)(input_layer)

    noise_input = tf.keras.layers.Input((unify_dim,))
    reshaped_noise_layer = tf.keras.layers.Reshape((1, unify_dim))(noise_input)

    concatenated_layer = tf.keras.layers.concatenate([reshaped_noise_layer, embedding_layer], axis=1)
    a_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True))(concatenated_layer)
    b_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True))(a_rnn_layer)
    c_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size))(b_rnn_layer)
    output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')(c_rnn_layer)
    model = tf.keras.models.Model(inputs=[input_layer, noise_input], outputs=[output_layer])

    # model.summary()
    assert model.output_shape == (None, vocab_size)
    return model


a_generator = generator()
# print(a_generator([tf.random.uniform((1, 64), minval=0, maxval=vocab_size - 1, dtype=tf.dtypes.int32,), tf.random.normal((1, 64))]))


def discriminator():
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Embedding(8192, unify_dim),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size)),
    #     tf.keras.layers.Dense(1),
    # ])
    input_layer = tf.keras.layers.Input((64,))
    embedding_layer = tf.keras.layers.Embedding(8192, unify_dim)(input_layer)

    one_hot_input = tf.keras.layers.Input((vocab_size, ))
    one_hot = tf.keras.layers.Dense(unify_dim)(one_hot_input)
    dense_layer = tf.keras.layers.Reshape((1, unify_dim))(one_hot)

    concatenated_layer = tf.keras.layers.concatenate([embedding_layer, dense_layer], axis=1)

    a_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True))(concatenated_layer)
    b_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size * 2, return_sequences=True))(a_rnn_layer)
    c_rnn_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_size))(b_rnn_layer)
    output_layer = tf.keras.layers.Dense(1)(c_rnn_layer)
    model = tf.keras.models.Model(inputs=[input_layer, one_hot_input], outputs=[output_layer])

    # model.summary()
    assert model.output_shape == (None, 1)
    return model


a_discriminator = discriminator()
a_discriminator_output = tf.random.uniform((1, 64), minval=0, maxval=vocab_size - 1, dtype=tf.dtypes.int32,)
# but remember add generator output to end of sequences in pool
# print(a_discriminator(a_discriminator_output))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './divar_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=a_generator,
                                 discriminator=a_discriminator)

BUFFER_SIZE = 1000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 64
num_examples_to_generate = 16

# # We will reuse this seed overtime (so it's easier)
# # to visualize progress in the animated GIF)
# seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(problem_space, solution_space, padded):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        last_words = a_generator([problem_space, noise], training=True)
        # TODO last_words = tf.nn.softmax_cross_entropy_with_logits(last_words, tf.reshape(last_words._targets, [-1]))

        real_output = a_discriminator([problem_space, solution_space], training=True)
        fake_output = a_discriminator([problem_space, last_words], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, a_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, a_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, a_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, a_discriminator.trainable_variables))

    # print(readout.format(epoch + 1, loss_values, metricsAcc.result() * 100))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for txt_batch in dataset:
            t = time.time()
            padding = np.random.randint(1, 63)
            solution_space = tf.one_hot(txt_batch[:, padding], vocab_size)
            problem_space = tf.concat([txt_batch[:, :padding], tf.zeros((txt_batch.shape[0], txt_batch.shape[1] - padding), dtype=tf.int32)], axis=1)
            train_step(problem_space, solution_space, padding)
            print(time.time() - t)
        # TODO Produce images for the GIF as we go
        # display.clear_output(wait=True)
        # generate_and_save_images(a_generator,
        #                          epoch + 1,
        #                          seed)

        # Save the model
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # generate_and_save_images(a_generator, epochs, seed)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

dir_name = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(dir_name, 'corpus.npy')):
    s = 0
    from pymongo import MongoClient
    ads = MongoClient()['divar']['ads']
    ads = [ad['description'] for ad in ads.find({})]
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(os.path.dirname(dir_name), "corpus.model"))
    vectors = []
    for text in ads:
        s += (len(sp.encode_as_ids(text)) - 60.686823009780205) ** 2
        vectors.append(sp.encode_as_ids(text)[:64])
        vectors[-1].extend([0] * (64 - len(vectors[-1])))

    np.save(os.path.join(dir_name, 'corpus.npy'), np.array(vectors).astype(np.int32))
    print(s / len(ads))

corpus = np.load(os.path.join(dir_name, 'corpus.npy'))
train_dataset = tf.data.Dataset.from_tensor_slices(corpus).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset, EPOCHS)
