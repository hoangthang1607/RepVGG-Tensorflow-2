import os
import time

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from repvgg import create_RepVGG_A0, repvgg_model_convert

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGENET_TRAINSET_SIZE = 1281167
IMG_SIZE = 64


def train_transform(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tfa.image.rotate(image, tf.random.uniform(shape=(), maxval=1))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.math.divide(image, 255.0)
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant(
        [0.229, 0.224, 0.225]
    )
    return image, label


def test_transform(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.math.divide(image, 255.0)
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant(
        [0.229, 0.224, 0.225]
    )
    return image, label


def main():
    train_model = create_RepVGG_A0()
    train_model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))

    ds = tfds.load("imagenet_resized/64x64", as_supervised=True)
    epochs = 120
    batch_size = 256
    lr = 0.1
    # weight_decay = 1e-4
    logdir = "logs_64x64"

    ds_train = (
        ds["train"]
        .shuffle(5 * batch_size)
        .map(train_transform, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    ds_test = (
        ds["validation"]
        .map(test_transform, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=epochs * IMAGENET_TRAINSET_SIZE // batch_size,
    )

    train_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.SGD(learning_rate=lr_decayed_fn,),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ],
    )

    train_model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0),
        ],
    )

    # check time for evaluation
    start_ts_eval = time.time()
    train_model.evaluate(ds_test)
    print(
        f"Evaluation of trained model takes: {time.time() - start_ts_eval:.4f}s"
    )
    train_model.save_weights(os.path.join(logdir, "RepVGG_train_phase"))
    deploy_model = repvgg_model_convert(
        train_model,
        build_func=create_RepVGG_A0,
        save_path=os.path.join(logdir, "RepVGG_test_phase"),
        image_size=(IMG_SIZE, IMG_SIZE, 3),
    )
    start_ts_eval = time.time()
    deploy_model.evaluate(ds_test)
    print(
        f"Evaluation of converted model takes: {time.time() - start_ts_eval:.4f}s"
    )

    # check convergence
    x = tf.random.uniform((batch_size, IMG_SIZE, IMG_SIZE, 3))
    train_y = train_model(x)
    deploy_y = deploy_model(x)
    print(tf.reduce_mean(tf.math.square(train_y - deploy_y)))


if __name__ == "__main__":
    main()
