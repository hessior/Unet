import numpy as np
from skimage.transform import resize, rotate, EuclideanTransform, warp
from skimage.util import crop

def data_aug(inpu, mask):
    '''inpu: tensor [bch, h, w, c],
       output: tensor [bch*some h, w, c]
    '''
    bch, h, w, c = inpu.shape
    # shuffle the data
    np.random.seed(100)
    np.random.shuffle(inpu)
    np.random.seed(100)
    np.random.shuffle(mask)    
    
    # crop the data
    crop_size = [((5,5),(8,8)), ((10,10),(15,15)), ((15,15),(20,20)), ((20,20),(30,30)), ((30,30),(40,40))]
    crop_inpu = np.zeros([bch//4, h, w, c])
    crop_mask = np.zeros([bch//4, h, w, c])
    for i in range(bch//4):
        idx = np.random.choice(5)
        crop_inpu[i,:,:,0] = resize(crop(inpu[i,:,:,0], crop_size[idx], copy=True), [h,w])
        crop_mask[i,:,:,0] = resize(crop(mask[i,:,:,0], crop_size[idx], copy=True), [h,w])
    
    # rotate the data
    rota_inpu = np.zeros([bch//4, h, w, c])
    rota_mask = np.zeros([bch//4, h, w, c])
    for i in range(bch//4):
        rota_inpu[i,:,:,0] = rotate(inpu[i,:,:,0], i%360)
        rota_mask[i,:,:,0] = rotate(mask[i,:,:,0], i%360)
    
    # translate data
    trans_size = [(1,1), (2,3), (3,4), (4,6)]
    trans_inpu = np.zeros([bch//4, h, w, c])
    trans_mask = np.zeros([bch//4, h, w, c])
    for i in range(bch//4):
        idx = np.random.choice(4)
        trans_inpu[i,:,:,0] = warp(inpu[i,:,:,0], EuclideanTransform(translation=trans_size[idx]))
        trans_mask[i,:,:,0] = warp(mask[i,:,:,0], EuclideanTransform(translation=trans_size[idx]))
    
    # flip data
    flip_inpu = np.zeros([bch//4, h, w, c])
    flip_mask = np.zeros([bch//4, h, w, c])
    for i in range(bch//4):
        idx = np.random.choice(2)
        flip_inpu[i,:,:,0] = np.flip(inpu[i,:,:,0],idx) 
        flip_mask[i,:,:,0] = np.flip(mask[i,:,:,0],idx)

    #print(inpu.shape, crop_inpu.shape, rota_inpu.shape, trans_inpu.shape)
    inpu = np.concatenate([inpu, crop_inpu, rota_inpu, trans_inpu, flip_inpu], axis=0)
    mask = np.concatenate([mask, crop_mask, rota_mask, trans_mask, flip_mask], axis=0)
    
    # shuffle again
    np.random.seed(100)
    np.random.shuffle(inpu)
    np.random.seed(100)
    np.random.shuffle(mask)
    
    return inpu, mask

if __name__ == "__main__":
    fake_img = np.random.normal(size=[16,160,256,1])
    fake_mask = np.random.normal(size=[16,160,256,1])
    a,b = data_aug(fake_img, fake_mask)
