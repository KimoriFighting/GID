import numpy as np
import cv2
import os

def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array,
    1 - mask,
    0 - background

    Returns run length as string formated
    '''
    pixels = img.T.flatten()  # 转置后看图像
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# 这个是用来解码train.csv中的Encoded Pixels的
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):  # 进行恢复
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def files_mask2rle(path):
    '''
    批量将mask转为rlu
    :param path:
    :return:
    '''
    files = os.listdir(path)
    csv = open(r'predict.csv', 'w')
    csv.writelines("filename,w h,rle编码\n")
    for file in files:
        fp = os.path.join(path, file)
        img = cv2.imread(fp)
        w, h = img.shape[1::-1]
        img = img[:, :, 0]
        img = img // 255
        result = mask2rle(img)
        csv.writelines("{},{} {},{}\n".format(file, w, h, result))


def files_rle2mask(csv, save_path):
    '''
    批量将rle转为mask
    :param csv:
    :return:
    '''

    csv = open(csv, 'r')
    line = csv.readline()
    for line in csv.readlines():
        arrs = line.split(',')
        name = arrs[0]
        w, h = list(map(int, arrs[1].split(' ')))
        rle = arrs[2]
        image = rle_decode(rle, (h, w))
        image = image * 255
        cv2.imwrite(os.path.join(save_path, name), image)


if __name__ == '__main__':
    # files_rle2mask(r'D:\Study\game\medical\train\mask.csv', r'D:\Study\game\medical\train\mask')
    files_mask2rle(r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/test/predict')



    # files_rle2mask(r"mask.csv", r'D:\document\experiment\data\data_med\ccf\data_med4\train')






