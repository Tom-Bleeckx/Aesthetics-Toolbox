import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F

_device = None

def _get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

################################ helper functions #####################################


def resize_and_add_ImageNet_mean(img):
    ### resize img to desired dimension
    img = resize(img, [512,512], order=1) ### Not the same "resize" function as the old Caffe code, which leads to different results depending on the extent of the resizing. 
    ### normalize with image_net mean
    img = img  - np.array([104.00698793 , 116.66876762 , 122.67891434])
    ### add new additional axis and return
    return img


def _prepare_input(input_img):
    """Prepare a single image for CNN processing (channel reorder, resize, normalize)."""
    img = input_img[:,:,(2,1,0)].astype(np.float32)  ## Caffe Net used different channel orders
    img = resize_and_add_ImageNet_mean(img)
    return img


def conv2d(input_img, kernel, bias):
    """Process a single image through AlexNet conv1."""
    input_img = _prepare_input(input_img)
    
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float().to(_get_device())
    kernel_tensor = torch.from_numpy(kernel).permute(3, 2, 0, 1).float().to(_get_device())
    bias_tensor = torch.from_numpy(bias).float().to(_get_device())
    
    with torch.no_grad():
        out = F.conv2d(input_tensor, kernel_tensor, bias=bias_tensor, stride=4)
        out = F.relu(out)
        
    output_data = out[0].cpu().numpy()
    return output_data


def conv2d_batch(input_imgs, kernel, bias):
    """Process multiple images through AlexNet conv1 in a single GPU call.
    
    Args:
        input_imgs: list of RGB numpy arrays
        kernel: AlexNet conv1 kernel
        bias: AlexNet conv1 bias
    Returns:
        list of output feature maps, each (out_channels, H, W)
    """
    prepared = [_prepare_input(img) for img in input_imgs]
    
    batch = np.stack(prepared, axis=0)
    input_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(_get_device())
    kernel_tensor = torch.from_numpy(kernel).permute(3, 2, 0, 1).float().to(_get_device())
    bias_tensor = torch.from_numpy(bias).float().to(_get_device())
    
    with torch.no_grad():
        out = F.conv2d(input_tensor, kernel_tensor, bias=bias_tensor, stride=4)
        out = F.relu(out)
    
    # Split back into individual results
    return [out[i].cpu().numpy() for i in range(len(input_imgs))]


def max_pooling (resp, patches ):
    (i_filters, ih, iw) = resp.shape
    max_pool_map = np.zeros((patches,patches,i_filters))
    patch_h = ih/float(patches)
    patch_w = iw/float(patches)

    for h in range(patches):
        for w in range(patches):
            ph = h*patch_h
            pw = w*patch_w
            patch_val = resp[:, int(ph):int(ph+patch_h), int(pw):int(pw+patch_w)]
            max_pool_map[h, w, :] = np.max(patch_val, axis=(1, 2))

    max_pool_map_sum = np.sum(max_pool_map, axis=2)
    normalized_max_pool_map = max_pool_map / max_pool_map_sum[:,:,np.newaxis]

    return max_pool_map, normalized_max_pool_map


def get_differences(max_pooling_map_orig, max_pooling_map_flip):
    assert(max_pooling_map_orig.shape == max_pooling_map_flip.shape)
    sum_abs = np.sum(np.abs(max_pooling_map_orig - max_pooling_map_flip))
    sum_max = np.sum(np.maximum(max_pooling_map_orig, max_pooling_map_flip))
    return 1.0 - sum_abs / sum_max


###################### Variances ####################################################


def CNN_Variance(normalized_max_pool_map, kind):
    '''
    Calculates the 'variability' or 'sparseness' QIP 
    
    Input: Takes the CNN-features of the first layer of an AlexNet as input
    Output: CNN_Variance, Sparseness or Variability
    
    Usage:
    Import Image from PIL    
    
    img_rgb = np.asarray(Image.open( path_to_image_file ).convert('RGB')) 
    
    patches = 12 # default is 12 for variability and 22 for sparseness 
    kind = 'variability' or 'sparseness'
    
    [kernel,bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)
    resp_scipy = CNN_qips.conv2d(img_rgb, kernel, bias)
    _, normalized_max_pooling_map_Variability = CNN_qips.max_pooling (resp_scipy, patches=patches )
    variability = CNN_qips.CNN_Variance (normalized_max_pooling_map_Variability , kind=kind )
    '''
    
    result = 0
    if kind == 'sparseness':
        result =  np.var( normalized_max_pool_map)
    elif kind == 'variability':
        result =  np.median(np.var(normalized_max_pool_map , axis=(0,1)))
    else:
        raise ValueError("Wrong input for kind of CNN_Variance. Use sparseness or variability")
    return result



################### Self-Similarity ################################

def CNN_selfsimilarity(histogram_ground, histogram_level):
    '''
    Calculates the 'CNN-based Self-similarity' QIP 
    
    Input: Takes the CNN-features of the first layer of an AlexNet as input
    Output: CNN-based Self-similarity
    
    Usage:
    Import Image from PIL    
    
    img_rgb = np.asarray(Image.open( path_to_image_file ).convert('RGB')) 
    
    [kernel,bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)
    resp_scipy = CNN_qips.conv2d(img_rgb, kernel, bias)
    _, normalized_max_pooling_map_8 = CNN_qips.max_pooling (resp_scipy, patches=8 )
    _, normalized_max_pooling_map_1 = CNN_qips.max_pooling (resp_scipy, patches=1 )
    cnn_self_sym = CNN_qips.CNN_selfsimilarity (normalized_max_pooling_map_1 , normalized_max_pooling_map_8 )
    '''
    
    ph, pw, n = histogram_level.shape
    hiks = []
    for ih in range(ph):
        for iw in range(pw):
            hiks.append( np.sum(np.minimum( histogram_ground, histogram_level[ih,iw])) )
    sesim = np.median(hiks)
    return sesim


################### CNN Symmetry ################################


def CNN_symmetry(input_img, kernel, bias):
    '''
    Calculates the 'CNN-feature-based Symmetry' QIP 
    
    Input: Takes the CNN-features of the first layer of an AlexNet as input
    Output: CNN-based Symmetry, left-rigth Symmetry, up-down Symmetry and left-right-up-down Symmetry
    
    Usage:
    Import Image from PIL    
    
    img_rgb = np.asarray(Image.open( path_to_image_file ).convert('RGB')) 
    
    [kernel,bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)
    sym_lr,sym_ud,sym_lrud = CNN_qips.CNN_symmetry(img_rgb, kernel, bias)
    '''
    
    # Prepare all 4 image variants
    img_lr = np.fliplr(input_img)
    img_ud = np.flipud(input_img)
    img_lrud = np.fliplr(img_ud)
    
    # Batch all 4 through the GPU in a single conv2d call
    resp_orig, resp_lr, resp_ud, resp_lrud = conv2d_batch(
        [input_img, img_lr, img_ud, img_lrud], kernel, bias
    )
    
    max_pooling_map_orig, _ = max_pooling(resp_orig, patches=17)
    
    max_pooling_map_lr, _ = max_pooling(resp_lr, patches=17)
    sym_lr = get_differences(max_pooling_map_orig, max_pooling_map_lr)
    
    max_pooling_map_ud, _ = max_pooling(resp_ud, patches=17)
    sym_ud = get_differences(max_pooling_map_orig, max_pooling_map_ud)

    max_pooling_map_lrud, _ = max_pooling(resp_lrud, patches=17)
    sym_lrud = get_differences(max_pooling_map_orig, max_pooling_map_lrud)
    
    return sym_lr, sym_ud, sym_lrud
