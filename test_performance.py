import time
import cProfile
import pstats
import os
import shutil

import QIP_machine_script

def setup():
    test_results_dir = './test_results/'
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)
    os.makedirs(test_results_dir, exist_ok=True)
    
    single_image_dir = './test_single_image'
    if os.path.exists(single_image_dir):
        shutil.rmtree(single_image_dir)
    os.makedirs(single_image_dir, exist_ok=True)
    
    src = './images/LogoDesign EAJ final.png'
    if os.path.exists(src):
        shutil.copy(src, single_image_dir)
    
    QIP_machine_script.results_path = test_results_dir
    QIP_machine_script.datasets = [['test_results.csv', single_image_dir]]

if __name__ == "__main__":
    setup() 
    profiler = cProfile.Profile()
    profiler.enable()
    QIP_machine_script.main()
    profiler.disable()
    
    with open('profile_stats.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('tottime')
        stats.print_stats(30)
