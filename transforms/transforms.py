import random
import numpy as np
from scipy.ndimage import rotate
from scipy import ndimage
import paddle
class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [DXHXW].

    Args:
        transforms (list): A list contains data pre-processing or augmentation.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): It is 3D (DxHxW).
            label (np.ndarray): It is 3D (DxHxW).

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if im is None:
            raise ValueError('None the image ')

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        if label is None:
            return (im,)
        else:
            return (im, label)


class RandomHorizontalFlip:
    """
     Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    Flip an image horizontally with a certain probability.

    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        assert im.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if random.random() < self.prob:
            if im.ndim == 3:
                im = np.flip(im,2)
                if label is not None:
                    label = np.flip(label,2)
            else:
                channels = [np.flip(im[c], 2) for c in range(im.shape[0])]
                im = np.stack(channels, axis=0)
                if label is not None:
                    channels = [np.flip(label[c], 2) for c in range(label.shape[0])]
                    label = np.stack(channels, axis=0)

        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomVerticalFlip:
    """
     Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    Flip an image vertically with a certain probability.

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        assert im.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if random.random() < self.prob:
            if im.ndim == 3:
                im = np.flip(im,1)
                if label is not None:
                    label = np.flip(label,1)
            else:
                channels = [np.flip(im[c], 1) for c in range(im.shape[0])]
                im = np.stack(channels, axis=0)
                if label is not None:
                    channels = [np.flip(label[c], 1) for c in range(label.shape[0])]
                    label = np.stack(channels, axis=0)

        if label is None:
            return (im, )
        else:
            return (im, label)


class Resize3D:
    """
    resample an image.
    Args:
        target_size (list|tuple, optional): The target size of image. Default: (32,256,256).

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
    """
    def __init__(self, target_size=(32,256,256), model='constant',order=1):
        self.model = model
        self.order=order

        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 3:
                raise ValueError(
                    '`target_size` should include 3 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.3D (DxHxW) or 4D (CxDxHxW)
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label),

        Raises:
            TypeError: When the 'img' type is not numpy.
            ValueError: When the length of "im" shape is not 3.
        """

        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if im.ndim == 3:
            desired_depth = depth = self.target_size[0]
            desired_width = width = self.target_size[1]
            desired_height = height = self.target_size[2]

            current_depth = im.shape[0]
            current_width = im.shape[1]
            current_height = im.shape[2]

            depth = current_depth / desired_depth
            width = current_width / desired_width
            height = current_height / desired_height
            depth_factor = 1 / depth
            width_factor = 1 / width
            height_factor = 1 / height

            im = ndimage.zoom(im, (depth_factor,width_factor, height_factor), order=self.order,mode=self.model)
            if label is not None:
                label = ndimage.zoom(label, (depth_factor,width_factor, height_factor), order=0,mode='nearest', cval=0.0)

        else:
            channels = [ndimage.zoom(im[c], (depth_factor,width_factor, height_factor), order=self.order,mode=self.model) for c
                        in range(im.shape[0])]
            im = np.stack(channels, axis=0)
            if label is not None:
                channels = [ndimage.zoom(label[c], label, (depth_factor,width_factor, height_factor), order=0,mode='nearest', cval=0.0) for c
                        in range(label.shape[0])]
                label = np.stack(channels, axis=0)

        if label is None:
            return (im, )
        else:
            return (im, label)

class RandomRotate:
    """
    Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self,  max_rotation=25, axes=None, mode='constant', order=0, **kwargs):
        
        self.max_rotation = max_rotation
        self.mode = mode
        self.order = order

    def __call__(self, im,label=None):
        axis = (2,1) 
        if self.max_rotation >0:
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            if im.ndim == 3:
                im = rotate(im, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
                if label is not None:
                    label = rotate(label, angle, axes=axis, reshape=False, order=self.order, mode='nearest', cval=0.0)
            else:
                channels = [rotate(im[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                            in range(im.shape[0])]
                im = np.stack(channels, axis=0)
                if label is not None:
                    channels = [rotate(label[c], angle, axes=axis, reshape=False, order=self.order, mode='nearest', cval=0.0) for c
                            in range(label.shape[0])]
                    label = np.stack(channels, axis=0)
        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomContrast:
    """
    Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, alpha=(0.2, 1.6), mean=0.0, prob=0.5, **kwargs):
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.prob = prob

    def __call__(self, im,label=None):
        
        if random.random() < self.prob:
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (im - self.mean)
            im = np.clip(result, -1, 1)

        if label is None:
            return (im, )
        else:
            return (im, label)


class Normalize:
    """
    Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, ww, wc, **kwargs):
        self.min_value = int(wc - (ww/2))
        self.max_value = int(wc + (ww/2))
        self.value_range = self.max_value - self.min_value

    def __call__(self, im, label = None):
        norm_0_1 = (im - self.min_value) / self.value_range
        im =  np.clip(2 * norm_0_1 - 1, -1, 1)

        if label is None:
            return (im, )
        else:
            return (im, label)

class ToTensor:
    """
    Converts a given input numpy.ndarray into paddle.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims=True, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, im,label=None):
        assert im.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and im.ndim == 3:
            im = np.expand_dims(im, axis=0)
            # if label is not None:
            #     label = np.expand_dims(label, axis=0)

        im = paddle.to_tensor(im.astype(dtype=self.dtype))
        if label is not None:
            label = paddle.to_tensor(label.astype(dtype='int32'))

        if label is None:
            return (im, )
        else:
            return (im, label)