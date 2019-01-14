import cv2 as cv
import numpy as np
from PIL import Image

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import detect_faces

if __name__ == "__main__":
    for i in range(10):
        img_fn = 'images/{}_raw.jpg'.format(i)
        print('Loading image {}'.format(img_fn))
        raw = cv.imread(img_fn, True)
        img = Image.open(img_fn).convert('RGB')
        _, facial5points = detect_faces(img)
        facial5points = np.reshape(facial5points[0], (2, 5))
        crop_size = (224, 224)

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (224, 224)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
        dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
        cv.imwrite('images/{}_warped.jpg'.format(i), dst_img)
        img = cv.resize(raw, (224, 224))
        cv.imwrite('images/{}_img.jpg'.format(i), img)
