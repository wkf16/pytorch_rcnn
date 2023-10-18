import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
model.load_state_dict(torch.load("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))

# 加载图像
image_path = "sample.png"
image = Image.open(image_path).convert("RGB")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # 缩放图像
    transforms.ToTensor(),  # 将PIL图像或NumPy的ndarray转换为FloatTensor，并缩放像素值为[0., 1.]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 应用预处理并添加批次维度
image_tensor = transform(image).unsqueeze(0)

# 如果你的设备支持CUDA，可以把模型放到GPU上
model = model.cuda() if torch.cuda.is_available() else model

# 将模型设置为评估模式
model.eval()

with torch.no_grad():  # 禁用梯度计算
    image_list = [image_tensor[0].cuda() if torch.cuda.is_available() else image_tensor[0]]
    prediction = model(image_list)


# 获得预测结果
prediction = prediction[0]

# 提取预测的边界框、标签和得分
boxes = prediction['boxes']
labels = prediction['labels']
scores = prediction['scores']

# 转换图像张量为NumPy数组，并展示图像
image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
plt.imshow(image_np)
ax = plt.gca()

# 只展示得分大于某个阈值（例如0.8）的边界框
for box, label, score in zip(boxes, labels, scores):
    if score > 0.1:
        x, y, x2, y2 = box
        rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f"{label}: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

plt.show()