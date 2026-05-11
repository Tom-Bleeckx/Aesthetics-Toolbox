import numpy as np
from skimage.transform import rotate
from skimage.filters import threshold_otsu

########################################################################################
################################# Huebner Group ########################################
########################################################################################


def Balance(img_gray):
    '''
    Calculates the "Balance" QIP from Ronald Huebner Group
    
    Input: Takes a grayscale image in Pillow format as input. 
    Output: Balance QIP
    
    Usage:
    Load images like this:
        
    Import Image from PIL    
    
    img_gray = np.asarray(Image.open( path_to_image_file ).convert('L')) 
    Balance(img_gray)
    '''
    
    height, width = img_gray.shape

    hist = np.histogram(img_gray, bins=256, range=(0, 256))

    counts = hist[0]
    
    thres = 128

    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres-1:])  # in the Matlab code, the treshold value 126 is added twice. This programming error has hardly any effect on the results and has been adopted here in the Python code.
    
    if sum1 <= sum2:
        im_comp = 255 - img_gray
    else:
        im_comp = img_gray

    nall = np.sum(im_comp)  
    
    ## to avoid division 0
    if nall == 0:
        nall = 1

    # Horizontal balance
    w = width // 2
    
    s1 = np.sum(im_comp[:, :w], dtype=int)
    s2 = np.sum(im_comp[:, -w:], dtype=int)
    bh = (abs(s1 - s2) / nall) * 100  
    
    w2 = width // 4 # adding center row of w to middle area if w uneven 

    s1 = np.sum(im_comp[:, :w2], dtype=int)
    s2 = np.sum(im_comp[:, -w2:], dtype=int)

    bioh = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100 # %  inner-outer horizontal 

    # Vertical balance
    h = height // 2

    s1 = np.sum(im_comp[:h, :], dtype=int)
    s2 = np.sum(im_comp[-h:, :], dtype=int)
    bv = (abs(s1 - s2) / nall) * 100
    
    h2 = height // 4
    s1 = np.sum(im_comp[:h2, :], dtype=int)
    s2 = np.sum(im_comp[-h2:, :], dtype=int)

    biov = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Main diagonal and inner-outer (bottom right top left)
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bmd = (abs(s1 - s2) / nall) * 100

    prop = 1 / np.sqrt(2)
    b1 = height - int(height * prop)
    b2 = width - int(width * prop)
    s1 = np.sum(np.tril(im_comp, -b1), dtype=int)
    s2 = np.sum(np.triu(im_comp, b2), dtype=int)
    biomd = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    # Anti-diagonal and inner-outer (bottom right top left)
    im_comp = np.rot90(im_comp)
    s1 = np.sum(np.triu(im_comp, 1), dtype=int)
    s2 = np.sum(np.tril(im_comp, -1), dtype=int)
    bad = (abs(s1 - s2) / nall) * 100

    s1 = np.sum(np.tril(im_comp, -b2), dtype=int)
    s2 = np.sum(np.triu(im_comp, b1), dtype=int)
    bioad = (abs((nall - (s1 + s2)) - (s1 + s2)) / nall) * 100

    bs = (bh + bv + bioh + biov + bmd + biomd + bad + bioad) / 8

    return bs


def DCM(img_gray):
    '''
    Calculates the "DCM" QIP from Ronald Huebner Group
    
    Input: Takes a grayscale image in Pillow format as input. 
    Output: DCM QIP
    
    Usage:
    Load images like this:
        
    Import Image from PIL    
    
    img_gray = np.asarray(Image.open( path_to_image_file ).convert('L')) 
    DCM(img_gray)
    '''
    
    height, width = img_gray.shape

    hist = np.histogram(img_gray, bins=256, range=(0, 256))
    counts = hist[0]
       
    thres = 128

    sum1 = sum(counts[:thres])
    sum2 = sum(counts[thres:])
    
    if sum1 <= sum2:
        im_comp = 255 - img_gray  # Invert image
    else:
        im_comp = img_gray

    nall = np.sum(im_comp)
    if nall == 0: nall = 1
    
    # Vectorized Horizontal balance point
    col_sums = np.sum(im_comp, axis=0, dtype=float)
    Rh = np.round(np.sum(col_sums * np.arange(width)) / nall) + 1
    Rhnorm = Rh / width
    
    # Vectorized Vertical balance point
    row_sums = np.sum(im_comp, axis=1, dtype=float)
    Rv = np.round(np.sum(row_sums * np.arange(height)) / nall) + 1
    Rvnorm = Rv / height

    htmp = 0.5 - Rhnorm
    vtmp = 0.5 - Rvnorm

    dist = np.sqrt(htmp ** 2 + vtmp ** 2)
    rdist = (dist / 0.5) * 100

    return rdist, htmp, vtmp


def Mirror_symmetry(img_gray):
    '''
    Calculates the "Mirror symmetry" QIP from Ronald Huebner Group
    
    Input: Takes a grayscale image in Pillow format as input. 
    Output: Mirror symmetry QIP
    '''

    level = threshold_otsu(img_gray)
    BW = img_gray <= level
    height, width = BW.shape

    # Vertical axis of reflection (horizontal symmetry)
    h2 = height // 2
    n1_h = h2 - 1 if h2 > 1 else 1
    BW_top = BW[:h2, :]
    BW_bottom = np.flipud(BW[height-h2:height, :])
    weight_h = (1 + np.arange(h2) / n1_h)[:, np.newaxis]
    Sh = np.sum(BW_top * BW_bottom * weight_h) * (2 / (3 * width * h2))

    # Horizontal axis of reflection (vertical symmetry)
    w2 = width // 2
    n1_w = w2 - 1 if w2 > 1 else 1
    BW_left = BW[:, :w2]
    BW_right = np.fliplr(BW[:, width-w2:width])
    weight_w = (1 + np.arange(w2) / n1_w)[np.newaxis, :]
    Sv = np.sum(BW_left * BW_right * weight_w) * (2 / (3 * height * w2))

    if width == height:
        # Diagonal symmetries (Squares only)
        # Major diagonal
        idx_i, idx_j = np.triu_indices(height, k=1)
        BW_upper = BW[idx_i, idx_j]
        BW_lower = BW[idx_j, idx_i]
        # Weighting for diagonal is complex to vectorize perfectly with loop parity, 
        # but we can use the same logic: weight = 1 + (j+1)/n where n is pixels until diagonal (which is i)
        weights_md = 1 + (idx_j + 1) / idx_i
        Smd = np.sum(BW_upper * BW_lower * weights_md) * (2 / (3 * height * (width - 1) / 2))

        # Minor diagonal
        BW_rot = np.rot90(BW)
        BW_upper_r = BW_rot[idx_i, idx_j]
        BW_lower_r = BW_rot[idx_j, idx_i]
        Sad = np.sum(BW_upper_r * BW_lower_r * weights_md) * (2 / (3 * height * (width - 1) / 2))

        ms = ((Sh + Sv + Smd + Sad) / 4) * 100
    else:
        ms = ((Sh + Sv) / 2) * 100

    return ms


def Homogeneity(img_gray):  
    '''
    Calculates the "Homogeneity" QIP from Ronald Huebner Group
    '''
    hbins, vbins = 10, 10
    height, width = img_gray.shape
    
    hist = np.histogram(img_gray, bins=256, range=(0, 256))
    counts = hist[0]
    if np.sum(counts[:128]) <= np.sum(counts[127:]):
        im = 255 - img_gray
    else:
        im = img_gray
    
    BW = im > threshold_otsu(im)

    # Use block reduction/reshaping to avoid loops
    hinc, vinc = width // hbins, height // vbins
    
    # Handle residuals by cropping to multiple of bins first, 
    # then adding back the edges if necessary to match original logic
    BW_core = BW[:vinc*vbins, :hinc*hbins]
    x = BW_core.reshape(vbins, vinc, hbins, hinc).sum(axis=(1, 3))
    
    # Add residuals to the last rows/cols as per original logic
    if height > vinc * vbins:
        x[-1, :] += BW[vinc*vbins:, :hinc*hbins].reshape(-1, hbins, hinc).sum(axis=(0, 2))
    if width > hinc * hbins:
        x[:, -1] += BW[:vinc*vbins, hinc*hbins:].reshape(vbins, vinc, -1).sum(axis=(1, 2))
    if height > vinc * vbins and width > hinc * hbins:
        x[-1, -1] += BW[vinc*vbins:, hinc*hbins:].sum()
    
    all_sum = np.sum(x)
    if all_sum == 0: return 0
    
    # Entropy calculations
    def calc_entropy(probs, max_val):
        probs = probs[probs > 0]
        return (-np.sum(probs * np.log2(probs)) / np.log2(max_val)) * 100

    en_hori = calc_entropy(np.sum(x, axis=0) / all_sum, hbins)
    en_vert = calc_entropy(np.sum(x, axis=1) / all_sum, vbins)
    
    return (en_hori + en_vert) / 2

