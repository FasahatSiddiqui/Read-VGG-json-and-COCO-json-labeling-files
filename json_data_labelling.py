import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import json
from skimage.draw import polygon

def json_mask(file_name,json_filename,file_extension, dir):
  img = cv2.imread(os.path.join(dir,'train_data',file_name),1)
  mask = np.zeros((img.shape[0],img.shape[1]))
  
  with open(os.path.join(dir,json_filename)) as f:
    data = json.load(f)

  ##-------------------------------------VGG json file
  if file_extension == 'VGG':
    for i in range(len(data['2.jpg']['regions'])):
      if data['2.jpg']['regions'][str(i)]['region_attributes']['label'] == 'leaf':
        x_points = data['2.jpg']['regions'][str(i)]['shape_attributes']['all_points_x']
        y_points = data['2.jpg']['regions'][str(i)]['shape_attributes']['all_points_y']
        x=np.round(np.array(x_points))
        y=np.round(np.array(y_points))
        rr, cc = polygon(y, x)
        mask[rr, cc] = 120
      if data['2.jpg']['regions'][str(i)]['region_attributes']['label'] == 'background':
        x_points = data['2.jpg']['regions'][str(i)]['shape_attributes']['all_points_x']
        y_points = data['2.jpg']['regions'][str(i)]['shape_attributes']['all_points_y']
        x=np.round(np.array(x_points))
        y=np.round(np.array(y_points))
        rr, cc = polygon(y, x)
        mask[rr, cc] = 255

    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imsave(dir+'/label_img_vgg.tiff',mask, cmap ='jet')
  
  ##--------------------------------COCO json file
  elif file_extension == 'COCO':
    from skimage.draw import polygon
    for i in range(len(data['annotations'])):
      if data['categories'][(data['annotations'][i]['category_id']-1)]['name'] == 'leaf':
        points = data['annotations'][i]['segmentation']
        points = np.array(points)
        points=np.transpose(points)
        points=np.round(points)
        x=[]
        y=[]
        for j in range (points.shape[0]):
          if j%2==0:
            x.append(points[j])
          else:
            y.append(points[j])
        rr, cc = polygon(y, x)
        mask[rr, cc] = 120
      if data['categories'][(data['annotations'][i]['category_id']-1)]['name'] == 'background':
        points = data['annotations'][i]['segmentation']
        points = np.array(points)
        points=np.transpose(points)
        points=np.round(points)
        x=[]
        y=[]
        for j in range (points.shape[0]):
          if j%2==0:
            x.append(points[j])
          else:
            y.append(points[j])
        rr, cc = polygon(y, x)
        mask[rr, cc] = 255

    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imsave(dir+'/label_img_coco.tiff',mask, cmap ='jet')


dir=os.getcwd() 
json_mask('.jpg','.json','COCO', dir)