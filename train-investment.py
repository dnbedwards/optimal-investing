# Adapted from TensorFlow Wide & Depp Tutorial

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt

_CSV_COLUMNS = [
    'Symbol', 'Date', 'Sector', 'Industry', 'MarketCapitalization', 'Beta', 'DividendYield', 'EarningsYield',
    'EbitYield', 'EbitdaYield', 'BookValueYield', 'CashflowYield', 'ReturnOnAssets', 'RevenueYield', 'DebtYield',
    'PriceChange'
]

_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''],
                        tf.constant([0], dtype=tf.int64), tf.constant([0], dtype=tf.float32),
                        tf.constant([0], dtype=tf.float32), tf.constant([0], dtype=tf.float32),
                        tf.constant([0], dtype=tf.float32), tf.constant([0], dtype=tf.float32),
                        tf.constant([0], dtype=tf.float32), tf.constant([0], dtype=tf.float32),
                        tf.constant([0], dtype=tf.float32), tf.constant([0], dtype=tf.float32),
                        tf.constant([0], dtype=tf.float32), tf.constant([0], dtype=tf.float32)]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/tmp/optimal_investing',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=20, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='../data-extraction/17q4-training.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='../data-extraction/17q4-test.csv',
    help='Path to the test data.')

parser.add_argument(
    '--training_increments', type=int, default=1, help='Train different sized batches for a useful training graph'
)

parser.add_argument(
    '--predict_file', type=str, default=''
)

_NUM_EXAMPLES = {
    'train': 2720,
    'validation': 1165,
    'predict': 478
}


def build_model_columns():
    # Continuous columns
    market_capitalization = tf.feature_column.numeric_column('MarketCapitalization',
                                                             normalizer_fn = lambda x: (x - 4000000000) / 14800000000)
    beta = tf.feature_column.numeric_column('Beta')
    dividend_yield = tf.feature_column.numeric_column('DividendYield')
    earnings_yield = tf.feature_column.numeric_column('EarningsYield')
    ebit_yield = tf.feature_column.numeric_column('EbitYield')
    ebitda_yield = tf.feature_column.numeric_column('EbitdaYield')
    book_value_yield = tf.feature_column.numeric_column('BookValueYield')
    cashflow_yield = tf.feature_column.numeric_column('CashflowYield')
    return_on_assets = tf.feature_column.numeric_column('ReturnOnAssets')
    revenue_yield = tf.feature_column.numeric_column('RevenueYield')
    debt_yield = tf.feature_column.numeric_column('DebtYield')

    # education = tf.feature_column.categorical_column_with_vocabulary_list(
    #    'education', [
    #        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    #        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    #        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    # hashing columns:
    sector = tf.feature_column.categorical_column_with_hash_bucket(
        'Sector', hash_bucket_size=100)
    industry = tf.feature_column.categorical_column_with_hash_bucket(
        'Industry', hash_bucket_size=100)

    # Transformations.
    # age_buckets = tf.feature_column.bucketized_column(
    #    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    deep_columns = [
        market_capitalization,
        beta,
        dividend_yield,
        earnings_yield,
        #ebit_yield,
        #ebitda_yield,
        book_value_yield,
        cashflow_yield,
        return_on_assets,
        revenue_yield,
        #debt_yield,
        tf.feature_column.indicator_column(sector),
        #tf.feature_column.indicator_column(industry),
    ]

    return deep_columns


def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    deep_columns = build_model_columns()

    # Weights 1
    #hidden_units = [100, 75, 50, 25]

    # Weights 2
    #hidden_units = [50, 35, 25, 12]

    # Weights 3
    #hidden_units = [50, 25, 10]

    # Weights 4
    hidden_units = [20, 10]

    # Weights 5
    #hidden_units = [40, 20, 10]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    return tf.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1)
    )


def input_fn(data_file, num_epochs, shuffle, batch_size, total_size):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
            '%s not found.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('PriceChange')
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.take(total_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=total_size)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def plot(x, training_cost, test_cost):
    fig, ax = plt.subplots()
    ax.plot(x, training_cost, color='k', label='Training error')
    ax.plot(x, test_cost, color='r', label='Test error')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')


def plot_batches(x, training_cost, test_cost):
    fig, ax = plt.subplots()
    ax.plot(x, training_cost, color='k', label='Training error')
    ax.plot(x, test_cost, color='r', label='Test error')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Cost')


def map_lines_to_symbols(file_name):
    with open(file_name) as f:
        content = f.readlines()
    return map(lambda line: line.split(',')[0], content)


def main(unused_argv):
    if len(FLAGS.predict_file) == 0:
        sys.exit(1)
        sizes = []
        training = []
        test = []

        for i in range(FLAGS.training_increments):
            train_size = int((i + 1) * (_NUM_EXAMPLES["train"] / FLAGS.training_increments))

            # Clean up the model directory if present
            shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
            model = build_estimator(FLAGS.model_dir)

            # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
            x = []
            y = []
            z = []

            for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
                model.train(input_fn=lambda: input_fn(
                    FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size, train_size))

                train_results = model.evaluate(input_fn=lambda: input_fn(
                    FLAGS.train_data, 1, False, FLAGS.batch_size, train_size))

                results = model.evaluate(input_fn=lambda: input_fn(
                    FLAGS.test_data, 1, False, FLAGS.batch_size, _NUM_EXAMPLES['validation']))

                # Display evaluation metrics
                print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
                print('-' * 60)

                x.append((n + 1) * FLAGS.epochs_per_eval)
                y.append(train_results["average_loss"])
                z.append(results["average_loss"])

                for key in sorted(results):
                    print('%s: %s' % (key, results[key]))

            if FLAGS.training_increments == 1:
                plot(x, y, z)
                #plt.show()
                current_time = time.time()
                plt.savefig("./training-graph-%d" % current_time)
            else:
                sizes.append(train_size)
                training.append(y[-1])
                test.append(z[-1])

        if FLAGS.training_increments > 1:
            plot_batches(sizes, training, test)
            current_time = time.time()
            plt.savefig("./training-size-graph-%d" % current_time)

    else:
        print('Symbol, Date, PredictedMovement')
        model = build_estimator(FLAGS.model_dir)
        predictions = model.predict(input_fn=lambda: input_fn(
            FLAGS.predict_file, 1, False, 1, _NUM_EXAMPLES['predict']
        ))
        symbols = map_lines_to_symbols(FLAGS.predict_file)

        for i, p in enumerate(predictions):
            print("%s,%f" % (symbols[i], p['predictions'][0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
