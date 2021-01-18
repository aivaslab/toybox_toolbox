import os
import conf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from PIL import Image


def get_bbox_df():
    return pd.read_csv('toybox_fps1_1080p_rot_bbox.csv')


def get_img_df():
    def get_info(p):
        base = os.path.basename(p)
        info, fmt_fr, _ = base.split('.')
        ca, no, _, tr = info.split('_')
        _, fr = fmt_fr.split('_')
        return p, ca, int(no), tr, int(fr)

    files = glob(f'{conf.PROJ_ROOT}/toybox_example/1080p/*.jpeg')
    img_df = pd.DataFrame(list(map(get_info, files)), columns=['path', 'ca', 'no', 'tr', 'fr'])
    return img_df


def plt_bbox_over_img(img_p, l, t, w, h):
    img = np.array(Image.open(img_p))
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((l, t), w, h, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    plt.show()


def main():
    bbox_df = get_bbox_df()
    img_df = get_img_df()
    plt_df = pd.merge(img_df, bbox_df, on=['ca', 'no', 'tr', 'fr'], how='left')

    img_row = plt_df.iloc[0]
    plt_bbox_over_img(img_row.path, img_row.left, img_row.top, img_row.width, img_row.height)


if __name__ == '__main__':
    main()

