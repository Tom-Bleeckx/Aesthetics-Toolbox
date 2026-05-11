#Import the required Libraries
import numpy as np
from PIL import Image
from skimage import color
import os
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path

### custom import
from AT import balance_qips, CNN_qips, color_and_simple_qips, edge_entropy_qips, fourier_qips, fractal_dimension_qips, PHOG_qips

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################################## set image paths and results.csv ##########################

### set path to save the results csv files
results_path = 'Results'


### each entry is a pair of the name of the csv file and the path to the image folder, you can enter several datasets/pairs
datasets = [
            [LOAD.results.csv'  , r'Photos\test'],   
            ]

####################################### set wanted QIPs to 'True', otherwise "False"  #########################

check_dict = {
    'Image size (pixels)': True,
    'Aspect ratio': True,
    'RMS contrast': True,
    'Luminance entropy': True,
    'Complexity': True,
    'Edge density': True,
    'Color entropy': True,
    'means RGB': True,
    'means Lab': True,
    'means HSV': True,
    'std RGB': True,
    'std Lab': True,
    'std HSV': True,
    'Mirror symmetry': True,
    'DCM': True,
    'Balance': True,
    'left-right': True,
    'up-down': True,
    'left-right & up-down': True,
    'Slope Redies': True,
    'Slope Spehar': True,
    'Slope Mather': True,
    'Sigma': True,
    '2-dimensional': True,
    '3-dimensional': True,
    'PHOG-based': True,
    'CNN-based': True,
    'Anisotropy': True,
    'Homogeneity': True,
    '1st-order': True,
    '2nd-order': True,
    'Sparseness': True,
    'Variability': True,
}

#######################################################################################################

Image.MAX_IMAGE_PIXELS = 1e10

dict_of_multi_measures = {
    'means RGB': ['mean R channel', 'mean G channel', 'mean B channel (RGB)'],
    'means Lab': ['mean L channel', 'mean a channel', 'mean b channel (Lab)'],
    'means HSV': ['mean H channel', 'mean S channel', 'mean V channel'],
    'std RGB': ['std R channel', 'std G channel', 'std B channel'],
    'std Lab': ['std L channel', 'std a channel', 'std b channel (Lab)'],
    'std HSV': ['std H channel', 'std S channel', 'std V channel'],
    'DCM': ['DCM distance', 'DCM x position', 'DCM y position'],
}

dict_full_names_QIPs = {
    'left-right': 'CNN symmetry left-right',
    'up-down': 'CNN symmetry up-down',
    'left-right & up-down': 'CNN symmetry left-right & up-down',
    '2-dimensional': '2D Fractal dimension',
    '3-dimensional': '3D Fractal dimension',
    'Sigma': 'Fourier sigma',
    'PHOG-based': 'Self-similarity (PHOG)',
    'CNN-based': 'Self-similarity (CNN)',
    '1st-order': '1st-order EOE',
    '2nd-order': '2nd-order EOE',
}

def custom_round(num):
    if num is None: return None
    if num < 1:
        scientific_notation = "{:e}".format(num)
        e_val = scientific_notation[-2:]
        return np.round(num, 3 + int(e_val))
    return np.round(num, 3)

class LazyImageHandler:
    def __init__(self, file_path, kernel=None, bias=None):
        self.file_path = file_path
        self.kernel = kernel
        self.bias = bias
        self._img_rgb = None
        self._img_lab = None
        self._img_hsv = None
        self._img_gray = None
        self._cnn_resp = None
        self._phog_results = None
        self._edge_entropy_results = None
        self._cnn_sym_results = None
        self._fourier_results = None

    @property
    def rgb(self):
        if self._img_rgb is None:
            self._img_rgb = np.asarray(Image.open(self.file_path).convert('RGB'))
        return self._img_rgb

    @property
    def lab(self):
        if self._img_lab is None:
            self._img_lab = color.rgb2lab(self.rgb)
        return self._img_lab

    @property
    def hsv(self):
        if self._img_hsv is None:
            self._img_hsv = color.rgb2hsv(self.rgb)
        return self._img_hsv

    @property
    def gray(self):
        if self._img_gray is None:
            self._img_gray = np.asarray(Image.open(self.file_path).convert('L'))
        return self._img_gray

    @property
    def cnn_resp(self):
        if self._cnn_resp is None:
            self._cnn_resp = CNN_qips.conv2d(self.rgb, self.kernel, self.bias)
        return self._cnn_resp

    @property
    def phog_results(self):
        if self._phog_results is None:
            self._phog_results = PHOG_qips.PHOGfromImage(self.rgb, section=2, bins=16, angle=360, levels=3, re=-1, sesfweight=[1,1,1])
        return self._phog_results

    @property
    def edge_entropy_results(self):
        if self._edge_entropy_results is None:
            self._edge_entropy_results = edge_entropy_qips.do_first_and_second_order_entropy_and_edge_density(self.gray)
        return self._edge_entropy_results

    @property
    def cnn_sym_results(self):
        if self._cnn_sym_results is None:
            self._cnn_sym_results = CNN_qips.CNN_symmetry(self.rgb, self.kernel, self.bias)
        return self._cnn_sym_results

    @property
    def fourier_results(self):
        if self._fourier_results is None:
            self._fourier_results = fourier_qips.fourier_redies(self.gray, bin_size=2, cycles_min=10, cycles_max=256)
        return self._fourier_results

def get_qip_registry():
    return {
        'means RGB': lambda h: color_and_simple_qips.mean_channels(h.rgb),
        'means Lab': lambda h: color_and_simple_qips.mean_channels(h.lab),
        'means HSV': lambda h: [color_and_simple_qips.circ_stats(h.hsv)[0], *color_and_simple_qips.mean_channels(h.hsv)[1:]],
        'std RGB': lambda h: color_and_simple_qips.std_channels(h.rgb),
        'std Lab': lambda h: color_and_simple_qips.std_channels(h.lab),
        'std HSV': lambda h: [color_and_simple_qips.circ_stats(h.hsv)[1], *color_and_simple_qips.std_channels(h.hsv)[1:]],
        'Color entropy': lambda h: color_and_simple_qips.shannonentropy_channels(h.hsv[:,:,0]),
        '1st-order': lambda h: h.edge_entropy_results[0],
        '2nd-order': lambda h: h.edge_entropy_results[1],
        'Edge density': lambda h: h.edge_entropy_results[2],
        'Luminance entropy': lambda h: color_and_simple_qips.shannonentropy_channels(h.lab[:,:,0]),
        'Image size (pixels)': lambda h: color_and_simple_qips.image_size(h.rgb),
        'Aspect ratio': lambda h: color_and_simple_qips.aspect_ratio(h.rgb),
        'left-right': lambda h: h.cnn_sym_results[0],
        'up-down': lambda h: h.cnn_sym_results[1],
        'left-right & up-down': lambda h: h.cnn_sym_results[2],
        'Sparseness': lambda h: CNN_qips.CNN_Variance(CNN_qips.max_pooling(h.cnn_resp, patches=22)[1], kind='sparseness'),
        'Variability': lambda h: CNN_qips.CNN_Variance(CNN_qips.max_pooling(h.cnn_resp, patches=12)[1], kind='variability'),
        'CNN-based': lambda h: CNN_qips.CNN_selfsimilarity(CNN_qips.max_pooling(h.cnn_resp, patches=1)[1], CNN_qips.max_pooling(h.cnn_resp, patches=8)[1]),
        'Sigma': lambda h: h.fourier_results[0],
        'Slope Redies': lambda h: h.fourier_results[1],
        'Slope Spehar': lambda h: fourier_qips.fourier_slope_branka_Spehar_Isherwood(h.gray),
        'Slope Mather': lambda h: fourier_qips.fourier_slope_mather(h.rgb),
        'RMS contrast': lambda h: color_and_simple_qips.std_channels(h.lab)[0],
        'Balance': lambda h: balance_qips.Balance(h.gray),
        'DCM': lambda h: balance_qips.DCM(h.gray),
        'Mirror symmetry': lambda h: balance_qips.Mirror_symmetry(h.gray),
        'Homogeneity': lambda h: balance_qips.Homogeneity(h.gray),
        '2-dimensional': lambda h: fractal_dimension_qips.fractal_dimension_2d(h.gray),
        '3-dimensional': lambda h: fractal_dimension_qips.fractal_dimension_3d(h.gray),
        'PHOG-based': lambda h: h.phog_results[0],
        'Complexity': lambda h: h.phog_results[1],
        'Anisotropy': lambda h: h.phog_results[2],
    }

def main():
    registry = get_qip_registry()
    [kernel, bias] = np.load(open("AT/bvlc_alexnet_conv1.npy", "rb"), encoding="latin1", allow_pickle=True)

    for csv_name, image_path in datasets:
        logger.info(f"Processing dataset: {csv_name}")
        full_csv_path = Path(results_path) / csv_name
        full_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Get already processed files
        existing_imgs = set()
        if full_csv_path.exists():
            try:
                existing_imgs = set(pd.read_csv(full_csv_path, usecols=['img_file'])['img_file'].tolist())
            except Exception as e:
                logger.warning(f"Could not read existing CSV: {e}")

        # Collect files to process
        file_paths = []
        for root, _, files in os.walk(image_path):
            for file in files:
                if file.replace(",", "_") not in existing_imgs:
                    file_paths.append(os.path.join(root, file))

        if not file_paths:
            logger.info("No new files to process.")
            continue

        results_batch = []
        batch_size = 50

        for file_dir in tqdm(file_paths):
            try:
                handler = LazyImageHandler(file_dir, kernel, bias)
                file_name = os.path.basename(file_dir).replace(",", "_")
                
                row = {'img_file': file_name}
                for key, enabled in check_dict.items():
                    if enabled:
                        res = registry[key](handler)
                        
                        if key in dict_of_multi_measures:
                            for i, sub_key in enumerate(dict_of_multi_measures[key]):
                                row[sub_key] = custom_round(res[i])
                        else:
                            col_name = dict_full_names_QIPs.get(key, key)
                            row[col_name] = custom_round(res)
                
                results_batch.append(row)

                if len(results_batch) >= batch_size:
                    df = pd.DataFrame(results_batch)
                    df.to_csv(full_csv_path, mode='a', index=False, header=not full_csv_path.exists())
                    results_batch = []

            except Exception as e:
                logger.error(f"Error processing {file_dir}: {e}")

        # Final batch
        if results_batch:
            df = pd.DataFrame(results_batch)
            df.to_csv(full_csv_path, mode='a', index=False, header=not full_csv_path.exists())

if __name__ == "__main__":
    main()
