import os
import sys

import seaborn
from keras.layers import GlobalMaxPooling2D

sys.path.insert(0, os.getcwd())  # add current working directory to pythonpath

import numpy as np
import pandas as pd
import warnings
import argparse
import gc
from sklearn.utils import class_weight
import tensorflow as tf
import matplotlib.pyplot as plt

tf_config = tf.compat.v1.ConfigProto
# set_session(tf.Session(config=tf_config))
from tensorflow.keras import applications, callbacks, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import config
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, top_k_accuracy_score

K.set_image_data_format('channels_last')


def model_builder(hp):
    """
    :param hp: kt.HyperParameters()
    :return: hp_learning_rate, shear_range
    """
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    dropout = hp.Boolean("dropout")

    return hp_learning_rate


def create_model(backbone='mobilenet'):
    """Build a CNN model with transfer learning from well-known architectures
    :param backbone: str, backbone architecture to use, can be mobilenet, inceptionV3, or resnet50
    :return model: Keras model
    :return img_size: size of input image for the model
    :return preprocessing_function: function to preprocess image for the model
    """

    if backbone == 'mobilenet':
        img_size = 224
        base_model = applications.mobilenet.MobileNet(include_top=False, weights='imagenet',
                                                      input_shape=(img_size, img_size, 3),
                                                      pooling=None)
        preprocessing_function = applications.mobilenet.preprocess_input
    elif backbone == 'inceptionV3':
        img_size = 299
        base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                           input_shape=(img_size, img_size, 3),
                                                           pooling=None)
        preprocessing_function = applications.inception_v3.preprocess_input
    elif backbone == 'resnet50':
        img_size = 299
        base_model = applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                  input_shape=(img_size, img_size, 3),
                                                  pooling=None)
        preprocessing_function = applications.resnet.preprocess_input
    else:
        raise ValueError('Backbone can only be mobilenet, inceptionV3, or resnet101.')

    for layer in base_model.layers:
        layer.trainable = False  # disable training of backbone
    x = base_model.output

    # Add classification head
    x = GlobalMaxPooling2D()(x)
    predictions = Dense(len(config.ALL_CAT), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, img_size, preprocessing_function


def get_prediction_score(y_true, y_predict_prob):
    """Evaluate predictions using different evaluation metrics.
    :param y_true: list, contains true label
    :param y_predict: list, contains predicted label
    :return scores: dict, evaluation metrics on the prediction
    """
    y_predict = np.argmax(y_predict_prob, axis=1)
    y_true_encoded = to_categorical(y_true)

    scores = {}
    scores[config.METRIC_ACCURACY] = accuracy_score(y_true, y_predict)
    scores[config.METRIC_F1_SCORE] = f1_score(y_true, y_predict, labels=None, average='macro', sample_weight=None)
    scores[config.METRIC_COHEN_KAPPA] = cohen_kappa_score(y_true, y_predict)
    scores[config.METRIC_CONFUSION_MATRIX] = confusion_matrix(y_true, y_predict)
    scores[config.METRIC_TOP_K] = top_k_accuracy_score(y_true, y_predict_prob, k=5)

    print('y_true:    ', y_true)
    print('y_predict: ', y_predict)

    return scores


def build_cnn_model(train_path, test_path,
                    backbone='mobilenet', batch_size=32, nb_epochs=100, lr=0.001,
                    save_path=None):
    """Train and evaluate CNN model
    :param train_path: path to train set,  which should have the below structure:
        train_path
            |---class_1
            |---class_2
            ...
            |---class_N
    :param test_path: path to test set,  which should have the below structure:
        test_path
            |---class_1
            |---class_2
            ...
            |---class_N
    :param backbone: str, contains backbone model name, which can be 'mobilenet', 'inceptionV3', 'resnet50'
    :param batch_size: int, batch size for model training
    :param nb_epochs: int, number of training epoches
    :param lr: float, learning rate
    :param save_path: path to save model
    :return model: fitted Keras model
    :return scores: dict, scores on test set for the fitted Keras model
    """
    # Create model
    model, img_size, preprocessing_function = create_model(backbone=backbone)
    # print(model.summary())

    # Prepare train and test data generator
    train_datagen = ImageDataGenerator(  # 변경
        preprocessing_function=preprocessing_function,
        rotation_range=90,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.5, 1.5],
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    y_train = train_generator.classes

    # Compute class weights
    weight_list = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {}
    for i in range(len(np.unique(y_train))):
        weight_dict[np.unique(y_train)[i]] = weight_list[i]

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

    # Callback list
    callback_list = []

    if save_path is not None:
        # save best model based on val_acc during training
        checkpoint = callbacks.ModelCheckpoint(os.path.join(save_path, backbone + 'weights.{epoch:02d}-'
                                                                                  '{val_top_k_categorical_'
                                                                                  'accuracy:.2f}.h5'),
                                               monitor='val_acc', # Can save best model only with val_acc available
                                               verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        callback_list.append(checkpoint)
        print('checkpoint', len(callback_list))

    # Train only classification head
    # optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])  # 변경: metrics = ['acc']
    model.fit(train_generator, validation_data=val_generator, epochs=nb_epochs,
                        class_weight=weight_dict, callbacks=callback_list, verbose=1)
    # model.fit(train_generator, validation_data=val_generator, epochs=nb_epochs,
    #           class_weight=weight_dict, verbose=1)

    # Train all layers
    for layer in model.layers:
        layer.trainable = True
    optimizer = optimizers.Adam(learning_rate=lr / 10)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
    history = model.fit(train_generator, validation_data=val_generator, epochs=nb_epochs,
                        class_weight=weight_dict, callbacks=callback_list, verbose=2)
    test_generator.reset()
    output = model.predict_generator(test_generator)
    print(test_generator.class_indices)
    # print("output", np.argmax(output, axis=1))

    # Evaluate the model
    print("Evaluate on test data")
    results = model.evaluate(test_generator, batch_size=16)
    print("test loss, test acc:", results)

    y_true = test_generator.classes
    y_predict_prob = model.predict(test_generator)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable the warning on f1-score with not all labels
        scores = get_prediction_score(y_true, y_predict_prob)

    plot_outcome(history, scores, False)

    return model, scores


def plot_outcome(history, scores, plot_show):
    # Plot outcome
    acc = history.history['top_k_categorical_accuracy']
    val_acc = history.history['val_top_k_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 10])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    if plot_show:
        plt.show()
    plt.savefig(backbone + '_acc')

    plt.subplot(1, 1, 1)
    cm = scores[config.METRIC_CONFUSION_MATRIX]

    seaborn.heatmap(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(backbone + '_cm')

    if plot_show:
        plt.show()


if __name__ == '__main__':
    tf.random.set_seed(config.SEED)
    data_path = os.path.join(config.WORK_DIRECTORY, config.CAT_DATASET_FOLDER)
    print('data path:', data_path)
    cnn_model_names = ['inceptionV3', 'resnet50'] # 변경: mobilenet 제거
    batch_size = 16
    nb_epochs = 100
    lr = 0.002  # 변경
    save_path = config.WORK_DIRECTORY

    # parse parameters
    parser = argparse.ArgumentParser(description='Build CNN models')
    parser.add_argument('--data_path', help='A path to folder containing train and test datasets')
    parser.add_argument('--batch_size', type=int, help='Batch size for model training')
    parser.add_argument('--nb_epochs', type=int, help='Number of training epoches')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--save_path', help='A path to save fitted models')

    args = parser.parse_args()
    if args.data_path:
        data_path = args.data_path
    if args.batch_size:
        batch_size = args.batch_size
    if args.nb_epochs:
        nb_epochs = args.nb_epochs
    if args.lr:
        lr = args.lr
    if args.save_path:
        save_path = args.save_path

    # Get path to train and test sets
    train_path = os.path.join(data_path, config.TRAIN_FOLDER)
    val_path = os.path.join(data_path, config.VAL_FOLDER)
    test_path = os.path.join(data_path, config.TEST_FOLDER)

    # Make save_path
    if save_path is not None:
        os.makedirs(os.path.join(save_path, '../cnn_models'), exist_ok=True)

    # Build CNN models
    cnn_model_scores = []
    for backbone in cnn_model_names:
        model, scores = build_cnn_model(train_path, test_path,
                                        backbone=backbone, batch_size=batch_size, nb_epochs=nb_epochs, lr=lr,
                                        save_path=os.path.join(save_path, '../cnn_models'))
        cnn_model_scores.append(scores)
        print(backbone, scores)
        # model.save('entire_model/model_0519.h5')

        # force release memory
        K.clear_session()
        del model
        gc.collect()

    # Summarize model performance
    model_df = pd.DataFrame({'model': cnn_model_names,
                             config.METRIC_ACCURACY: [score[config.METRIC_ACCURACY] for score in cnn_model_scores],
                             config.METRIC_F1_SCORE: [score[config.METRIC_F1_SCORE] for score in cnn_model_scores],
                             config.METRIC_COHEN_KAPPA: [score[config.METRIC_COHEN_KAPPA] for score in
                                                         cnn_model_scores],
                             config.METRIC_CONFUSION_MATRIX: [score[config.METRIC_CONFUSION_MATRIX] for score in
                                                              cnn_model_scores],
                             config.METRIC_TOP_K: [score[config.METRIC_TOP_K] for score in cnn_model_scores]
                             })
    model_df = model_df[['model', config.METRIC_ACCURACY, config.METRIC_F1_SCORE, config.METRIC_COHEN_KAPPA,
                         config.METRIC_CONFUSION_MATRIX, config.METRIC_TOP_K]]
    model_df.to_csv(os.path.join(config.WORK_DIRECTORY, '../summary_cnn_model.csv'), index=False)
    model_df.sort_values(by=[config.METRIC_ACCURACY, config.METRIC_TOP_K, config.METRIC_F1_SCORE,
                             config.METRIC_COHEN_KAPPA],
                         ascending=False, inplace=True)
    print('Best model:\n' + str(model_df.iloc[0]))
