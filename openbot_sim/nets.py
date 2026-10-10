"""Neural-network definitions used by the training scripts.

This is a self-contained, minimal re-implementation of the network
architectures used by OpenBot (https://github.com/ob-f/OpenBot,
`policy/openbot/models.py`, Intel Intelligent Systems Lab, 2020).
The interface contract (image + command inputs, two wheel-command
outputs) is kept identical to the original so the resulting ``.tflite``
models are drop-in compatible with the OpenBot Android app.

Providing this file means the repository does **not** depend on the
OpenBot source tree at all.
"""

import tensorflow as tf


def create_cnn(
    width,
    height,
    depth,
    cnn_filters=(8, 12, 16, 20, 24),
    kernel_sz=(5, 5, 5, 3, 3),
    stride=(2, 2, 2, 2, 2),
    padding="same",
    activation="relu",
    conv_dropout=0,
    mlp_filters=(64, 16),
    mlp_dropout=0.2,
    bn=False,
):
    """Conv feature extractor: image -> feature vector."""
    input_shape = (height, width, depth)
    channel_dim = -1
    inputs = tf.keras.Input(shape=input_shape, name="img_input")

    x = inputs
    for i, f in enumerate(cnn_filters):
        x = tf.keras.layers.Conv2D(
            f,
            (kernel_sz[i], kernel_sz[i]),
            strides=(stride[i], stride[i]),
            padding=padding,
            activation=activation,
        )(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channel_dim)(x)
        if conv_dropout > 0:
            x = tf.keras.layers.Dropout(conv_dropout)(x)

    x = tf.keras.layers.Flatten()(x)
    for f in mlp_filters:
        x = tf.keras.layers.Dense(f, activation=activation)(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channel_dim)(x)
        if mlp_dropout > 0:
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

    return tf.keras.Model(inputs, x)


def create_mlp(in_dim, hidden_dim, out_dim, activation="relu", dropout=0.2, name="cmd"):
    """Small MLP branch used for the high-level command input."""
    model = tf.keras.Sequential(name="MLP")
    model.add(
        tf.keras.layers.Dense(hidden_dim, input_dim=in_dim, activation=activation, name=name)
    )
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(out_dim, activation=activation))
    return model


def cil_mobile(img_width, img_height, bn=True, policy="autopilot"):
    """CIL-Mobile-Cmd: the default OpenBot autopilot policy.

    Note: the original uses dropout=0.5. Our experiments show this is too
    strong for small datasets / weak steering signals and induces shortcut
    learning (see README). Pass ``dropout`` via a copy of this function if
    you want to change it; ``scripts/train.py`` exposes a ``--dropout`` flag.
    """
    if policy == "autopilot":
        mlp = create_mlp(1, 16, 16, dropout=0.5, name="cmd")
    elif policy == "point_goal_nav":
        mlp = create_mlp(3, 16, 16, dropout=0.5, name="goal")
    else:
        raise Exception("Unknown policy")

    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 64, 96, 128, 256),
        kernel_sz=(5, 3, 3, 3, 3),
        stride=(2, 2, 2, 2, 2),
        padding="same",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(128, 64),
        mlp_dropout=0.5,
        bn=bn,
    )

    combined = tf.keras.layers.concatenate([mlp.output, cnn.output])
    x = tf.keras.layers.Dense(64, activation="relu")(combined)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    return tf.keras.Model(name="cil_mobile", inputs=(cnn.input, mlp.input), outputs=x)


def simple(img_width, img_height, policy="autopilot"):
    """A minimal architecture: small CNN + command concat + dense head.

    No dropout / batch-norm. Used as the 'fixed' model in our experiments.
    """
    base = create_cnn(img_width, img_height, 3)
    img_in = tf.keras.Input((img_height, img_width, 3))
    cmd_in = tf.keras.Input((1,))
    feat = base(img_in)
    x = tf.keras.layers.Concatenate()([feat, cmd_in])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="linear")(x)
    return tf.keras.Model([img_in, cmd_in], out)
