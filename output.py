import os
import cv2
import numpy as np


def evaluate(model, concat=True, summary=True):
    if summary:
        model.summary()

    dir_original = "evaluation/original/"
    dir_output = "evaluation/output/"

    image_names = os.listdir(dir_original)

    for image_name in image_names:
        # Read in and format original image
        image = cv2.imread(f"{dir_original}{image_name}")
        input = image
        dim = image.shape

        # Reshape image
        image = np.reshape(image, (1, *dim)) / 255

        # Get Output
        output = np.array(model(image)[0])
        output = output * 255

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"evaluation/Combined/{image_name}",
                np.concatenate((input, output), axis=1),
            )


if __name__ == "__main__":
    import tensorflow as tf
    from keras.models import load_model

    # If uncommented, forces the use of the cpu or gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Hides tensorflow outputs and warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Prevents complete memory allocation of gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    model = load_model("weights/weights.h5")
    evaluate(model)
