import numpy as np

from keras.utils import Sequence


def create_center_mask(patch_size, center_size):
    mask = np.zeros(patch_size[:2])
    h, w = patch_size[:2]
    y_start = h // 2 - center_size[0] // 2
    x_start = w // 2 - center_size[1] // 2
    mask[y_start:y_start + center_size[0], x_start:x_start + center_size[1]] = 1

    return np.expand_dims(mask, axis=2)


def create_rnd_patch(img, patch_size, seed):
    radius = img.shape[0] / 2
    
    np.random.seed(seed)
    angle = np.random.randint(0, 359)
    distance = np.random.randint(0, radius - np.sqrt(np.sum(np.power(patch_size, 2))))
    x_start, y_start = (int(radius + distance * np.cos(angle)), int(radius - distance * np.sin(angle)))
    
    return img[y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]


def prepare_patch(img, patch_size, mask, seed):
    patch = create_rnd_patch(img, patch_size, seed)
    patch = (1 - mask) * patch
    
    return patch.astype('float32') / 255.0


class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, imgs, patch_size, center_size, batch_size=32, shuffle=True):

        self.imgs = imgs
        self.patch_size = patch_size
        self.center_size = center_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.epoch = 0        
        self.on_epoch_end()
        
        self.mask = create_center_mask(patch_size, center_size)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.imgs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        x = self.__data_generation(indexes, [self.imgs[k] for k in indexes])

        return x, x

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.epoch += 1
        
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.seed(self.epoch)
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, indexes, batch):      
        """Augments or/and pretransforms data"""
        patches = [prepare_patch(img, self.patch_size, self.mask, (self.epoch * idx) % (2**32 - 1)) 
                   for idx, img in zip(indexes, batch)]

        return np.array(patches)