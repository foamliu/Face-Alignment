import cv2 as cv
import numpy as np

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector


def process(img, output_size):
    _, facial5points = detector.detect_faces(img)
    facial5points = np.reshape(facial5points[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    cv.imwrite('images/{}_mtcnn_aligned_{}x{}.jpg'.format(i, output_size[0], output_size[1]), dst_img)
    # img = cv.resize(raw, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)


if __name__ == "__main__":
    detector = MtcnnDetector()

    for i in range(10):
        filename = 'images/{}_raw.jpg'.format(i)
        print('Loading image {}'.format(filename))
        raw = cv.imread(filename)
        process(raw, output_size=(224, 224))
        process(raw, output_size=(112, 112))
