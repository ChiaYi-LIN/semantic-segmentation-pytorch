# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import glob
import os
import json
from PIL import Image
from print_utils import *

def get_mappings(root_dir, files, annot_name):
    pairs = []
    for f in files:
        f = f.replace(root_dir, '')
        img_f = f.replace(annot_name, 'leftImg8bit')
        img_f = img_f.replace('_labelTrainIds.png', '.png')
        if not os.path.isfile(root_dir + img_f):
            print_error_message('{} file does not exist. Please check'.format(root_dir + img_f))
            exit()
        img = Image.open(os.path.join(root_dir, img_f))
        segm = Image.open(os.path.join(root_dir, f))
        img_width, img_height = img.size
        segm_width, segm_height = segm.size
        assert(img_width == segm_width and img_height == segm_height)
        pairs.append({
            "fpath_img": img_f,
            "fpath_segm": f,
            "width": img_width,
            "height": img_height,
        })
    return pairs

def main(cityscapesPath, split):
    searchFine = os.path.join(cityscapesPath, "gtFine", split, "*", '*_labelTrainIds.png')
    filesFine = glob.glob(searchFine)
    filesFine.sort()

    if not filesFine:
        print_warning_message("Did not find any gtFine files. Please check root directory: {}.".format(cityscapesPath))
        fine_pairs = []
    else:
        print_info_message('{} files found for {} split'.format(len(filesFine), split))
        fine_pairs = get_mappings(cityscapesPath, filesFine, 'gtFine')

    if not fine_pairs:
        print_error_message('No pair exist. Exiting')
        exit()
    else:
        print_info_message('Creating train and val files.')
    f_name = split + '.odgt'
    with open(os.path.join(cityscapesPath, f_name), 'w') as jsonfile:
        for pair in fine_pairs:
            json.dump(pair, jsonfile)
            jsonfile.write('\n')
    print_info_message('{} created in {} with {} pairs'.format(f_name, cityscapesPath, len(fine_pairs)))

    if split == 'train':
        split_orig = split
        split = split + '_extra'
        searchCoarse = os.path.join(cityscapesPath, "gtCoarse", split, "*", '*_labelTrainIds.png')
        filesCoarse = glob.glob(searchCoarse)
        filesCoarse.sort()
        if not filesCoarse:
            print_warning_message("Did not find any gtCoarse files. Please check root directory: {}.".format(cityscapesPath))
            course_pairs = []
        else:
            print_info_message('{} files found for {} split'.format(len(filesCoarse), split))
            course_pairs = get_mappings(cityscapesPath, filesCoarse, 'gtCoarse')
        if not course_pairs:
            print_warning_message('No pair exist for coarse data')
            return
        else:
            print_info_message('Creating train and val files.')
        f_name = split_orig + '_coarse.odgt'
        with open(os.path.join(cityscapesPath, f_name), 'w') as jsonfile:
            for pair in course_pairs:
                json.dump(pair, jsonfile)
                jsonfile.write('\n')          
        print_info_message('{} created in {} with {} pairs'.format(f_name, cityscapesPath, len(course_pairs)))

if __name__ == '__main__':
    cityscapes_path = '../data/cityscapes/'
    main(cityscapes_path, "train")
    main(cityscapes_path, "val")