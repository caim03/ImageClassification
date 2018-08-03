import tensorflow as tf
import json
from dataset import Dataset

tf.logging.set_verbosity(tf.logging.INFO)


def build_model_fn(is_training, images, params):
    """ Compute logits of the model """

    # Reshape input in correct format
    out = tf.reshape(images, [-1, params['Height'], params['Width'], params['Channels']])  # [-1, 240, 320, 3]

    # Extracts features with first convolution layer
    out = tf.layers.conv2d(inputs=out, filters=96, kernel_size=[11, 11], strides=4, padding='same', activation=tf.nn.relu)

    # Normalization layer to avoid overfitting
    out = tf.nn.local_response_normalization(input=out, depth_radius=2, alpha=0.00002, beta=0.75, bias=1)

    # Reduce dimension with pooling layer
    out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='valid')

    # Extracts more specific features with second convolution layer
    out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[5, 5], strides=1, padding='same', activation=tf.nn.relu)

    # Normalization layer to avoid overfitting
    out = tf.nn.local_response_normalization(input=out, depth_radius=2, alpha=0.00002, beta=0.75, bias=1)

    # Reduce dimension with second pooling layer - output [60, 80, 16]
    out = tf.layers.max_pooling2d(inputs=out, pool_size=[3, 3], strides=2, padding='valid')

    # Third convolution layer
    out = tf.layers.conv2d(inputs=out, filters=384, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)

    # Fourth convolution layer
    out = tf.layers.conv2d(inputs=out, filters=384, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)

    # Five convolution layer
    out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu)

    # Flat output to use a fully connected MLP
    out = tf.layers.flatten(out)

    # Dense layer to get classification
    out = tf.layers.dense(inputs=out, units=4096, activation=tf.nn.relu)

    # Dropout layer to perform regularization
    #out = tf.layers.dropout(inputs=out, rate=0.5, training=is_training)

    # Dense layer to get classification
    out = tf.layers.dense(inputs=out, units=4096, activation=tf.nn.relu)

    # Dropout layer to perform regularization
    #out = tf.layers.dropout(inputs=out, rate=0.5, training=is_training)

    # Logits layer - 4 classes
    # Input Tensor Shape: [batch_size, 4096]
    # Output Tensor Shape: [batch_size, 4]
    logits = tf.layers.dense(inputs=out, units=4)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """ Model function defining graph operations """

    model_spec = {}

    is_training = (mode == 'train')
    labels = inputs['labels']
    images = inputs['images']

    # Define the model
    # Compute logits and predictions
    with tf.variable_scope('model', reuse=reuse):
        logits= build_model_fn(is_training, images, params)
        # Return the position in which the probability is max in softmax or sparse softmax (aka the class predicted)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
    precision, precision_op = tf.metrics.precision(labels=labels, predictions=predictions)
    recall, recall_op = tf.metrics.recall(labels=labels, predictions=predictions)

    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['accuracy_op'] = accuracy_op
    model_spec['precision'] = precision
    model_spec['precision_op'] = precision_op
    model_spec['recall'] = recall
    model_spec['recall_op'] = recall_op
    model_spec['labels'] = labels
    model_spec['pred'] = predictions

    # In this case we must find the best weight for our network
    if is_training:
        optimizer = tf.train.AdamOptimizer(params['LearningRate'])
        global_step = tf.train.get_or_create_global_step()

        if params['BatchNorm'] == 1:
            # Use Batch Normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)

        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        train_loss = tf.summary.scalar('train_loss', loss)
        train_accuracy = tf.summary.scalar('train_accuracy', accuracy_op)
        train_precision = tf.summary.scalar('train_precision', precision_op)
        train_recall = tf.summary.scalar('train_recall', recall_op)
        train_summary = tf.summary.merge([train_loss, train_accuracy, train_precision, train_recall])

        model_spec['train_op'] = train_op
        model_spec['train_summary'] = train_summary

    else:
        val_loss = tf.summary.scalar('val_loss', loss)
        val_accuracy = tf.summary.scalar('val_accuracy', accuracy_op)
        val_precision = tf.summary.scalar('val_precision', precision_op)
        val_recall = tf.summary.scalar('val_recall', recall_op)
        val_summary = tf.summary.merge([val_loss, val_accuracy, val_precision, val_recall])

        model_spec['val_summary'] = val_summary

    return model_spec


def main(argv=None):
    json_file = open('config.json', 'r')
    json_file = json.load(json_file)

    training_param = json_file['Training']
    validation_param = json_file['Validation']
    parameters = json_file['Parameters']

    training = Dataset(path=training_param, param=parameters)
    validation = Dataset(path=validation_param, param=parameters)

    train_set, train_label = training.get_next()
    val_set, val_label = validation.get_next()

    training_set = {
        'images': train_set,
        'labels': train_label
    }

    validation_set = {
        'images': val_set,
        'labels': val_label
    }

    epoch_num = parameters['NumEpoch']
    batch_size = parameters['BatchSize']
    train_batch_num = training.lenght/batch_size
    val_batch_num = validation.lenght/batch_size

    train_spec = model_fn(mode='train', inputs=training_set, params=parameters)
    val_spec = model_fn(mode='validation', inputs=validation_set, params=parameters, reuse=True)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    best_val_acc = 0

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_local)
        sess.run(training.init_iterator())
        sess.run(validation.init_iterator())

        train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
        val_writer = tf.summary.FileWriter("./logs/validation")

        for epoch in range(epoch_num):
            print("\nEpoch:{}\n".format(epoch))
            train_progress = tf.keras.utils.Progbar(train_batch_num)

            print("\nStart Training...\n")
            for step in range(train_batch_num):
                _, train_loss, train_acc, train_acc_op, train_prec, train_prec_op, train_rec, train_rec_op, train_summary = \
                    sess.run([train_spec['train_op'], train_spec['loss'], train_spec['accuracy'],
                            train_spec['accuracy_op'], train_spec['precision'], train_spec['precision_op'], train_spec['recall'], train_spec['recall_op'],
                            train_spec['train_summary']])

                train_progress.update(step, [("train_loss", train_loss), ("train_accuracy", train_acc_op), ("train_precision", train_prec_op),
                                             ("train_recall", train_rec_op)])

            print("\nStart Validation...\n")
            val_progress = tf.keras.utils.Progbar(val_batch_num)
            for step in range(val_batch_num):
                val_loss, val_acc, val_acc_op, val_prec, val_prec_op, val_rec, val_rec_op, val_summary = \
                    sess.run([val_spec['loss'], val_spec['accuracy'], val_spec['accuracy_op'],
                            val_spec['precision'], val_spec['precision_op'], val_spec['recall'], val_spec['recall_op'],
                            val_spec['val_summary']])

                print(sess.run(val_spec['labels']))
                print(sess.run(val_spec['pred']))

                val_progress.update(step, [("val_loss", val_loss), ("val_accuracy", val_acc_op), ("val_precision", val_prec_op),
                                           ("val_recall", val_rec_op)])

            train_writer.add_summary(train_summary, epoch)
            val_writer.add_summary(val_summary, epoch)

            if val_acc_op > best_val_acc:
                best_val_acc = val_acc_op
                saver.save(sess, "./logs/acc_{}".format(val_acc_op), global_step=epoch)
            print("\nBest val accuracy: {}".format(best_val_acc))

if __name__ == "__main__":
    tf.app.run()





