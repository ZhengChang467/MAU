import numpy as np
import torch
import cv2


def reshape_patch(img_tensor, patch_size):
    assert 4 == img_tensor.ndim
    seq_length = np.shape(img_tensor)[0]
    img_height = np.shape(img_tensor)[1]
    img_width = np.shape(img_tensor)[2]
    num_channels = np.shape(img_tensor)[3]
    a = np.reshape(img_tensor, [seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 3, 2, 4, 5])
    patch_tensor = np.reshape(b, [seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor


def reshape_patch_back_tensor(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    patch_narray = patch_tensor.detach().cpu().numpy()
    batch_size = np.shape(patch_narray)[0]
    seq_length = np.shape(patch_narray)[1]
    patch_height = np.shape(patch_narray)[2]
    patch_width = np.shape(patch_narray)[3]
    channels = np.shape(patch_narray)[4]
    img_channels = channels // (patch_size * patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                     patch_height, patch_width,
                                     patch_size, patch_size,
                                     img_channels])
    b = a.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                   patch_height * patch_size,
                                   patch_width * patch_size,
                                   img_channels])
    return img_tensor.permute(0, 1, 4, 2, 3)


def reshape_patch_tensor(img_tensor, patch_size):
    assert 4 == img_tensor.ndim
    seq_length = img_tensor.shape[0]
    img_height = img_tensor.shape[1]
    img_width = img_tensor.shape[2]
    num_channels = img_tensor.shape[3]
    a = torch.reshape(img_tensor, [seq_length,
                                   img_height // patch_size, patch_size,
                                   img_width // patch_size, patch_size,
                                   num_channels])
    b = a.permute((0, 1, 3, 2, 4, 5))
    patch_tensor = torch.reshape(b, [seq_length,
                                     img_height // patch_size,
                                     img_width // patch_size,
                                     patch_size * patch_size * num_channels])
    return patch_tensor.permute((0, 3, 1, 2))



