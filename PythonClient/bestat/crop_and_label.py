import cv2
import os
import random


source_rel = 'booth_v1'
target_rel = 'booth_v1_1_crop_and_labeled'


cwd = os.getcwd()

source = os.path.join(cwd, source_rel)
target = os.path.join(cwd, target_rel)
target_neg = os.path.join(target, 'negative')
target_pos = os.path.join(target, 'positive')

os.makedirs(target_neg, exist_ok=True)
os.makedirs(target_pos, exist_ok=True)

human_color = (6, 108, 153)  # BGR
item_color = (191, 105, 112)  # BGR

count = 0
for path, _, files in os.walk(source):

    # ignore intermediate folders.
    if len(_) > 0:
        continue

    folder_name = path.split('\\')[-1]
    _, level, human, item, *_ = folder_name.split('_')
    print('processing {}'.format(folder_name))

    x0, y0, x1, y１ = 0, 60, 640, 188

    for file in files:

        name, ext = os.path.splitext(file)
        suffix, postfix = name.split('_')

        if postfix not in ['rgbL']:
            continue
        if ext not in ['.jpg', '.jpeg', '.png']:
            continue

        img = cv2.imread(os.path.join(path, file))
        img_cropped = img[y0:y1, x0:x1, :]

        mask = cv2.imread(os.path.join(path, suffix + '_' + 'maskL.png'))
        mask_cropped = mask[y0:y1, x0:x1, :]


        target_class = target_neg

        def count_mask_pixels(mask, color):
            match = ((mask[:, :, 0] == color[0]) & (mask[:, :, 1] == color[1]) & (mask[:, :, 2] == color[2])).sum()
            return match

        # we exclude the image where the human('s hand) is reaching the top/sides since its pose is unnatural in the booth.
        w = mask_cropped.shape[1]
        boundary_human_pixels = count_mask_pixels(mask_cropped[:, 0:1, :], human_color) + \
            count_mask_pixels(mask_cropped[:, w-1:w, :], human_color) + \
            count_mask_pixels(mask_cropped[0:1, :, :], human_color)
        if boundary_human_pixels > 0:
            continue

        # we exclude the image with non-zero but few item pixels since such sample is 'semipositive'.
        item_pixels = count_mask_pixels(mask_cropped, item_color)
        if 0 < item_pixels <= 600:
            continue
        elif 600 < item_pixels:
            target_class = target_pos

        # we also exclude the image where the human('s hand) is not so appearent but the item is.
        human_pixels = count_mask_pixels(mask_cropped, human_color)
        if target_class == target_pos and human_pixels <= 2000:
            continue

        save_file = os.path.join(target_class, '{}_{}_{}_{:0>8}.jpg'.format(level, human, item, count))
        cv2.imwrite(save_file, img_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        count += 1