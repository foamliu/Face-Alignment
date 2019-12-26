import unittest

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from align_faces import REFERENCE_FACIAL_POINTS, get_reference_facial_points, warp_and_crop_face
from mtcnn.detector import MtcnnDetector

detector = MtcnnDetector()


def crop_test(image,
              facial5points,
              reference_points=None,
              output_size=(96, 112),
              align_type='similarity'):
    # dst_img = transform_and_crop_face(image, facial5points, coord5points, imgSize)

    dst_img = warp_and_crop_face(image,
                                 facial5points,
                                 reference_points,
                                 output_size,
                                 align_type)

    print('warped image shape: ', dst_img.shape)

    # swap BGR to RGB to show image by pyplot
    dst_img_show = dst_img[..., ::-1]

    plt.figure()
    plt.title(align_type + ' transform ' + str(output_size))
    plt.imshow(dst_img_show)
    plt.show()


class TestMethods(unittest.TestCase):
    image = None
    facial5points = None
    reference_5pts = None
    output_size = None

    def setUp(self):
        img_fn = 'images/0_img.jpg'
        self.image = cv.imread(img_fn)
        _, facial5points = detector.detect_faces(self.image)
        facial5points = np.reshape(facial5points[0], (2, 5))
        self.facial5points = facial5points

        default_square = True
        inner_padding_factor = 0
        outer_padding = (0, 0)
        output_size = (112, 112)

        self.reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        self.output_size = output_size

    def test_get_reference_facial_points(self):
        default_square = True
        inner_padding_factor = 0
        outer_padding = (0, 0)
        output_size = (112, 112)

        reference_5pts = get_reference_facial_points(output_size,
                                                     inner_padding_factor,
                                                     outer_padding,
                                                     default_square)

        print('--->reference_5pts:\n', reference_5pts)

        try:
            dft_5pts = np.array(REFERENCE_FACIAL_POINTS)
            plt.title('Default 5 pts')
            #        plt.axis('equal')
            plt.axis([0, 96, 112, 0])
            #        plt.xlim(0, 96)
            #        plt.ylim(0, 112)
            plt.scatter(dft_5pts[:, 0], dft_5pts[:, 1])

            plt.figure()
            plt.title('Transformed new 5 pts')
            #        plt.axis('equal')
            plt.axis([0, 112, 112, 0])
            #        plt.xlim(0, 224)
            #        plt.ylim(0, 224)
            plt.scatter(reference_5pts[:, 0], reference_5pts[:, 1])
            plt.show()
        except Exception as e:
            print('Exception caught when trying to plot: ', e)

        # self.assertEqual('foo'.upper(), 'FOO')

    def test_warp_and_crop_face(self):
        img_fn = 'images/0_img.jpg'
        # imgSize = [96, 112]; # cropped dst image size

        # facial points in cropped dst image
        #    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
        #                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];

        # facial points in src image
        # facial5points = [[105.8306, 147.9323, 121.3533, 106.1169, 144.3622],
        #                 [109.8005, 112.5533, 139.1172, 155.6359, 156.3451]];

        #    facial5points = np.array(facial5points)
        print('Loading image {}'.format(img_fn))
        image = cv.imread(img_fn, True)

        # for pt in src_pts[0:3]:
        #   cv.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), 1, 8, 0)

        # swap BGR to RGB to show image by pyplot
        image_show = image[..., ::-1]

        plt.figure()
        plt.title('src image')
        plt.imshow(image_show)
        plt.show()
        # self.assertTrue('FOO'.isupper())
        # self.assertFalse('Foo'.isupper())

    def test_default_crop_setting_with_similarity_transform(self):
        crop_test(self.image, self.facial5points)

    def test_default_crop_setting_with_cv2_affine_transform(self):
        crop_test(self.image, self.facial5points, align_type='cv2_affine')

    def test_default_crop_setting_with_default_affine_transform(self):
        crop_test(self.image, self.facial5points, align_type='affine')

    def test_default_square_crop_setting(self):
        # crop settings, set the region of cropped faces
        #    default_square = True
        #    inner_padding_factor = 0.25
        #    outer_padding = (0, 0)
        #    output_size = (224, 224)

        default_square = True
        inner_padding_factor = 0
        outer_padding = (0, 0)
        output_size = (112, 112)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)
        print('--->reference_5pts:\n', reference_5pts)

    def test_default_square_crop_setting_with_similarity_transform(self):
        crop_test(self.image, self.facial5points, self.reference_5pts, self.output_size)

    def test_default_square_crop_setting_with_affine_transform(self):
        crop_test(self.image, self.facial5points, self.reference_5pts, self.output_size,
                  align_type='cv2_affine')

    def test_default_square_crop_setting_with_default_affine_transform(self):
        crop_test(self.image, self.facial5points, self.reference_5pts, self.output_size,
                  align_type='affine')


if __name__ == '__main__':
    unittest.main()
