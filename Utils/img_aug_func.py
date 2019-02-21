# import cv2
from scipy.ndimage.interpolation    import map_coordinates
# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
from skimage import io
from scipy.ndimage.filters import gaussian_filter as gaussian
import time
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries
from random import shuffle


import colorsys

N = 1000
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list (map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

COLOR_LIST = [(int (rgb[0] * 255), int (rgb[1] * 255), int (rgb[2] * 255)) for rgb in RGB_tuples]
shuffle (COLOR_LIST)
COLOR_LIST [0] = (0, 0, 0)

def index2rgb (index):
    return COLOR_LIST[index]

def lbl2rgb (lbl):
    original_shape = np.squeeze (lbl).shape
    lbl = lbl.flatten ().tolist ()
    lbl = list (map (index2rgb, lbl))
    lbl = np.array (lbl).reshape (original_shape + (3,))
    return lbl


def read_im (paths):
    ret = []
    for path in paths:
        ret.append (io.imread (path))
    return ret

def random_reverse(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        rng = np.random.RandomState(seed)
        random_reverse = rng.randint(1,3)
    if random_reverse==1:
        reverse = image[::1,...]
    elif random_reverse==2:
        reverse = image[::-1,...]
    image = reverse
    return image

def random_square_rotate(image, seed=None):
    assert ((image.ndim == 2) | (image.ndim == 3))
    if seed:
        rng = np.random.RandomState(seed)        
    rot_k = rng.randint(0,4)
    rotated = image
    if image.ndim==2:
        rotated = np.rot90(image, rot_k, axes=(0,1))
    elif image.ndim==3:
        rotated = np.rot90(image, rot_k, axes=(1,2))
    image = rotated
    return image
            
# def random_elastic(image, seed=None):
#   assert ((image.ndim == 2) | (image.ndim == 3))
#   old_shape = image.shape

#   if image.ndim==2:
#       image = np.expand_dims(image, axis=0) # Make 3D

#   if seed:
#       np.random.seed(seed)

#   new_shape = image.shape
#   dimx, dimy = new_shape[1], new_shape[2]
#   size = np.random.randint(4,16) #4,32
#   ampl = np.random.randint(2, 5) #4,8
#   du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
#   dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
#   # Done distort at boundary
#   du[ 0,:] = 0
#   du[-1,:] = 0
#   du[:, 0] = 0
#   du[:,-1] = 0
#   dv[ 0,:] = 0
#   dv[-1,:] = 0
#   dv[:, 0] = 0
#   dv[:,-1] = 0
    
#   # Interpolate du
#   DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
#   DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
#   X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
#   indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
    
#   warped = image
#   for z in range(new_shape[0]): #Loop over the channel
#       # print z
#       imageZ = np.squeeze(image[z,...])
#       flowZ  = map_coordinates(imageZ, indices, order=0).astype(np.float32)

#       warpedZ = flowZ.reshape(image[z,...].shape)
#       warped[z,...] = warpedZ     
#   warped = np.reshape(warped, old_shape)
#   return warped

def random_gaussian_blur (image, n, seed=None):
    blured = []
    for i in range (n):
        x = np.random.randint (0, len (image))
        blured += [x]
        if not x in blured:
            image[x] = gaussian (image[x], sigma=1)
    return image

def random_blackout (image, n, randt, range_xy = (50, 256)):
    blacked = []
    for i in range (n):
        x = randt.randint (len (image))
        while (x in blacked):
            x = randt.randint (len (image))
        blacked += [x]
    for i in blacked:
        lenx = randt.randint (range_xy[0], range_xy[1])
        leny = randt.randint (range_xy[0], range_xy[1])
        x0 = randt.randint (image.shape[1] - lenx + 1)
        y0 = randt.randint (image.shape[2] - leny + 1)
        value = float (1.0 * randt.randint (255) / 255.0) 
        image[i, y0:y0+leny, x0:x0 + lenx] = value
    return image


def square_rotate(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))        
    rot_k = n
    rotated = image.copy()
    if image.ndim==2:
        rotated = np.rot90(image, rot_k, axes=(0,1))
    elif image.ndim==3:
        rotated = np.rot90(image, rot_k, axes=(1,2))
    image = rotated
    return image

def random_flip(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    random_flip = n
    if random_flip==1:
        flipped = image[...,::1,::-1]
        image = flipped
    elif random_flip==2:
        flipped = image[...,::-1,::1]
        image = flipped
    elif random_flip==3:
        flipped = image[...,::-1,::-1]
        image = flipped
    elif random_flip==4:
        flipped = image
        image = flipped
    return image

def reverse(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    random_reverse = n
    if random_reverse==1:
        reverse = image[::1,...]
    elif random_reverse==2:
        reverse = image[::-1,...]
    image = reverse
    return image

def erode_label (imgs, iterations=1):
    ret = []
    for img in imgs:
        bndr_map = 1 - find_boundaries (img)
        bndr_map = binary_erosion (bndr_map, iterations=iterations)
        ret.append (np.multiply (bndr_map, img).astype (img.dtype))
    return ret


def flip(image, n):
    assert ((image.ndim == 2) | (image.ndim == 3))
    random_flip = n
    if random_flip==1:
        flipped = image[...,::1,::-1]
        image = flipped
    elif random_flip==2:
        flipped = image[...,::-1,::1]
        image = flipped
    elif random_flip==3:
        flipped = image[...,::-1,::-1]
        image = flipped
    elif random_flip==4:
        flipped = image
        image = flipped
    return image

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

def apply_aug (block, labels, func, seed=None):
    return func (block, seed), func (labels, seed)  
