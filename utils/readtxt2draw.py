import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib

# 1. 指定中文字体（以 Windows 上的“SimHei”黑体为例）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

def smooth_curve(points, factor=0):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 读取.txt文件中的数据
Epoch = []
lr = []
Tr_loss = []
Val_loss = []
Acc = []
F1 = []
Iou = []
Presion = []
Recall = []
with open(r"C:/ss/wrjtlxj/xm/HRSICD-main/result/HRSICD_6/3333.txt", 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        values = line.split('\t')
        Epoch.append(float(values[0]))
        lr.append(float(values[1]))
        Tr_loss.append(float(values[2]))
        Val_loss.append(float(values[3]))
        Acc.append(float(values[4]))
        F1.append(float(values[5]))
        Iou.append(float(values[6]))
        Presion.append(float(values[7]))
        Recall.append(float(values[8]))

# 对训练损失和验证损失进行平滑处理
tr_loss = smooth_curve(Tr_loss)
val_loss = smooth_curve(Val_loss)

plt.figure(figsize=(10, 8))

# 绘制所有曲线，并设置不同颜色和标签
plt.plot(Epoch, lr, color='blue', label='Learning Rate')
plt.plot(Epoch, tr_loss, color='red', linestyle='--', label='Train Loss')
plt.plot(Epoch, val_loss, color='green', linestyle='-', label='Validation Loss')
plt.plot(Epoch, Acc, color='orange', label='Accuracy')
plt.plot(Epoch, F1, color='purple', label='F1 Score')
plt.plot(Epoch, Iou, color='brown', label='IoU')
plt.plot(Epoch, Presion, color='pink', label='Precision')
plt.plot(Epoch, Recall, color='cyan', label='Recall')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('各指标随Epoch变化趋势')
plt.legend()  # 显示图例
plt.grid(True)

plt.show()
