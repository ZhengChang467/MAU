from torchvision import transforms
from torch.utils.data import DataLoader
from core.data_provider.mm import MovingMNIST


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True,
                  is_shuffle=True):
    if is_training:
        num_workers = configs.num_workers
        root = data_train_path
    else:
        num_workers = 0
        root = data_test_path
    dataset = MovingMNIST(is_train=is_training,
                          root=root,
                          n_frames=20,
                          num_objects=[2])
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)
