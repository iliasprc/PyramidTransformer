# import glob
# import os
#
# # path = '/home/iliask/Desktop/ilias/datasets/GSL_test_data/GSL_in_the_wild/'
# # videos = sorted(glob.glob(f"{path}**/**"))
# # print(len(videos))
# # f = open('GSL_in_the_WILD.csv', 'w')
# # for i in videos:
# #     print(i)
# #     extension = i[-4:]
# #     filename = i[:-4].replace(';', '').replace('+', ' ').replace(',', '').replace('ΚΕΠ1-', '').replace('ΚΕΠ2-', '').replace('ΚΕΠ4-',
# #                                                                                                            '').replace(
# #         'ΚΕΠ5-', '').replace('ΚΕΠ3-', '').replace('ΥΓΕ1-','').replace('ΠΩΣ_','ΠΩΣ').replace('ΤΙ_','ΤΙ')
# #     print(filename + extension)
# #     name = filename.split('/')[-1]
# #     #os.rename(i,filename + extension)
# #     #name = name.replace('ΚΕΠ1-', '').replace('+', ' ').replace('-', ' ').replace(';', '')
# #     # print(name)
# #     f.write(f"{i.replace('/home/iliask/Desktop/ilias/datasets/GSL_test_data/', '')},{name},*\n")
#
#
#
# def load_csv_file()

#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
# import urllib.request
#
# # URLs for the zip files
# links = [
#     'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
#     'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
#     'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
# 	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
#     'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
# 	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
# 	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
#     'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
# 	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
# 	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
# 	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
# 	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
# ]
#
# for idx, link in enumerate(links):
#     fn = 'images_%02d.tar.gz' % (idx+1)
#     print('downloading'+fn+'...')
#     urllib.request.urlretrieve(link, fn)  # download the zip file
import os

import cv2
# path = path.replace('.png','.jpg')
# img = cv2.imread('/home/iliask/PycharmProjects/SLVTP/cxr8/images_01/images/00000001_000.png')
# img = cv2.resize(img,(300,300))
# cv2.imwrite('/home/iliask/PycharmProjects/SLVTP/cxr8/images_01/images/00000001_000.jpg',img)

import glob
path = '/home/iliask/PycharmProjects/SLVTP/cxr8/images_12/images/'
import os
if not os.path.exists(path.replace('images/', 'jpg_images/')):
    os.makedirs(path.replace('images/', 'jpg_images/'))

#pathnew = '/home/iliask/PycharmProjects/SLVTP/cxr8/images_01/jpg_images/'
images  = sorted(glob.glob(f'{path}**.png'))
print(images)

# path = '/home/iliask/PycharmProjects/SLVTP/cxr8/images_01/images/00000001_000.png'
#


for path in images:

    img = cv2.imread(path)
    #print(img.shape)
    assert img.shape ==(1024,1024,3)
    img = cv2.resize(img, (300, 300))
    pathnew = path.replace('.png', '.jpg').replace('images/', 'jpg_images/')
    cv2.imwrite(pathnew, img)
