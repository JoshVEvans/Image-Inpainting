import os

import matplotlib.pyplot as plt
import numpy as np
import cv2


def evaluate(model, concat=True, summary=True):
    if summary:
        model.summary()

    dir_original = "evaluation/original/"
    dir_input = "evaluation/input/"
    dir_output = "evaluation/output/"

    image_names = os.listdir(dir_input)

    for image_name in image_names:
        # Read in and format original image
        original = cv2.imread(f"{dir_original}{image_name}")
        image = cv2.imread(f"{dir_input}{image_name}")
        input = image
        dim = image.shape

        # Reshape image
        image = np.reshape(image, (1, *dim)) / 255

        # Get Output
        output = np.array(model(image)[0])
        output = output * 255

        # White Strip
        white = np.full(
            shape=(output.shape[0], output.shape[1] // 50, 3), fill_value=255
        )

        # Write Output
        cv2.imwrite(f"{dir_output}{image_name}", output)

        if concat:
            cv2.imwrite(
                f"evaluation/Combined/{image_name}",
                np.concatenate((original, white, input, white, output), axis=1),
            )

            f = plt.figure()

            img1 = f.add_subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(np.array(original), cv2.COLOR_BGR2RGB))

            img2 = f.add_subplot(1, 3, 2)
            plt.imshow(
                cv2.cvtColor(np.array(input), cv2.COLOR_BGR2RGB),
            )

            img3 = f.add_subplot(1, 3, 3)
            output = np.clip(output, 0, 255).astype(np.uint8)
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

            img1.set_xlabel("original")
            img1.xaxis.set_ticks([])
            img1.yaxis.set_ticks([])

            img2.set_xlabel(f"interpolated")
            img2.xaxis.set_ticks([])
            img2.yaxis.set_ticks([])

            img3.set_xlabel(f"network / {round(cv2.PSNR(original, output), 2)}dB")
            img3.xaxis.set_ticks([])
            img3.yaxis.set_ticks([])

            plt.savefig(f"evaluation/plot/{image_name}", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    import tensorflow as tf
    from create_sample import main
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
    main()
    evaluate(model)
