import neuroglancer
import numpy as np
import os
import cv2

ip = '10.112.127.227'  # or public IP of the machine for sharable display
port = 10000  
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer = neuroglancer.Viewer()

res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['nm', 'nm', 'nm'],
        scales=[20, 5, 5])


def image_to_array(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_array = np.array(gray_img)
    return img_array


def folder_to_array(folder_path):
    img_list = os.listdir(folder_path)
    img_arrays = []
    for img_name in img_list:
        img_path = os.path.join(folder_path, img_name)
        img_array = image_to_array(img_path)
        img_arrays.append(img_array)
    return img_arrays


er = folder_to_array('./er')
er_seg = np.array(er)
mito = folder_to_array('./mito')
mito_seg = np.array(mito)



def ngLayer(data, res, oo=[0,0,0], tt='segmentation'):
    return neuroglancer.LocalVolume(data, dimensions=res, volume_type=tt, voxel_offset=oo)

with viewer.txn() as s:
    s.layers.append(name='seg_er', layer=ngLayer(er_seg, res, tt='segmentation'))
    s.layers.append(name='seg_mito', layer=ngLayer(mito_seg, res, tt='segmentation'))

print(viewer)
