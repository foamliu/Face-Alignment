import cv2 as cv
import numpy as np
from PIL import Image

from mtcnn.detector import detect_faces
from warp_and_crop_face import warp_and_crop_face

if __name__ == "__main__":
    for i in range(10):
        img_fn = 'images/{}_img.jpg'.format(i)
        print('Loading image {}'.format(img_fn))
        image = cv.imread(img_fn, True)
        img = Image.open(img_fn).convert('RGB')
        _, facial5points = detect_faces(img)
        facial5points = np.reshape(facial5points[0], (2, 5))
        crop_size = (112, 112)

        dst_img = warp_and_crop_face(image,
                                     facial5points)
        cv.imwrite('images/{}_warped.jpg'.format(i), dst_img)
