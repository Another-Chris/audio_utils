import tensorflow as tf


class StatisticalPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=1)
        std = tf.math.sqrt(tf.math.reduce_variance(inputs, axis=1) + 1e-12)
        return tf.concat([mean, std], 1)


""" 2d base """


def get_base_2d(input_shape, dp_rate=0.2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(inputs)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 11), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 10))(x)
    x = tf.keras.layers.Dropout(dp_rate)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 11), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 5))(x)
    x = tf.keras.layers.Dropout(dp_rate)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(7, 11), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((5, 10))(x)

    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


""" 1d """
def get_base_1d(input_shape, dp_rate=0.2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1)))(inputs)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(dp_rate)(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Dropout(dp_rate)(x)

    x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dp_rate)(x)

    x = StatisticalPooling1D()(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def get_embeddings(base):
    inputs = base.inputs

    x = tf.keras.layers.Dense(units=256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def get_classifier(embedding, n_classes, activation="softmax"):
    inputs = embedding.inputs

    x = tf.keras.layers.Dense(n_classes)(embedding.output)
    x = tf.keras.layers.Activation(activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model


""" big models """
def get_resnet(input_shape, n_classes):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    inputs = model.input
    x = tf.keras.layers.GlobalMaxPooling2D()(model.output)
    x = tf.keras.layers.Dense(n_classes)(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def get_xception(input_shape, n_classes):
    model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    inputs = model.input
    x = tf.keras.layers.GlobalMaxPooling2D()(model.output)
    x = tf.keras.layers.Dense(n_classes)(x)
    # x = tf.keras.layers.Activation("sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


