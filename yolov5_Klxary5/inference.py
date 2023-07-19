from models.common import * 
import torch.onnx
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms.functional as T

checkpoint = torch.load(r'D:\desktop\项目\危险品检测\yolov5_Klxary5\runs\train\exp\weights\best.pt',map_location='cpu')
model = checkpoint["model"].float()
model.eval()

model.fuse()
model.model[-1].export = True

#在Python上进行Yolov5的推理

#读取图像转换到RGB
image = cv2.imread(r'data\images\P00001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#把图像进行缩放，长边缩放到网络大小，使得图像能够整数32
#这里直接使用warpAffine进行居中缩放
#一定要注意，推理使用时，由于tensorRT的网络分辨率是固定的，所以我们采用居中缩放
#Yolov5中的Detect，是使用最小图像，例如。310 * 640 的输入， 他的推理将会是320 * 640 结果是推理时图像大小可能不同（这在tensortRT中不允许）
#Yolov5在评估mAP时，使用的时最小图像+pad，也就是640的图回使用672尺寸，边缘多了一小圈
image_height, image_width = image.shape[:2]
scale = 640 / max(image_height, image_width)
offset_x = 320 - image_width * scale * 0.5
offset_y = 320 - image_height * scale *0.5

M = np.array([
    [scale, 0, offset_x],
    [0, scale, offset_y]
], dtype=np.float32)

image = cv2.warpAffine(image, M, (640, 640), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

#normalzie 除以255
# 如果提供的是float， to_tensor不会干啥，仅仅permute
# 如果同功的是int，to_tensor 会除以255，然后permute
image = T.to_tensor(image).unsqueeze(dim=0)

#Focus
# cat([
#     image[..., ::2, ::2],
#     image[..., 1::2, ::2],
#     image[..., ::2, 1::2],
#     image[..., 1::2, 1::2]
# ])
image = torch.cat([
    image[..., ::2, ::2],
    image[..., 1::2, ::2],
    image[..., ::2, 1::2],
    image[..., 1::2, 1::2]
],dim=1)


# 推理
with torch.no_grad():
    predict = model(image)


print("done")