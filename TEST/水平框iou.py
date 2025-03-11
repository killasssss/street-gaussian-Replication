import numpy as np  # 导入 NumPy 库

def single_iou(bbox1, bbox2):
    """
    计算两个水平矩形框（bbox）的 IoU（Intersection over Union）

    参数:
    bbox1, bbox2: list 或 array，格式为 [x1, y1, x2, y2]
                  - (x1, y1): 左上角坐标
                  - (x2, y2): 右下角坐标

    返回:
    IoU 值: 两个矩形框的交集面积 / 并集面积
    """

    # 计算两个框的交集（Intersection）区域的左上角和右下角坐标
    xleft = np.maximum(bbox1[0], bbox2[0])   # 交集区域左上角 x 坐标
    yleft = np.maximum(bbox1[1], bbox2[1])   # 交集区域左上角 y 坐标
    xright = np.minimum(bbox1[2], bbox2[2])  # 交集区域右下角 x 坐标
    yright = np.minimum(bbox1[3], bbox2[3])  # 交集区域右下角 y 坐标

    # 计算交集区域的宽度（w）和高度（h），并确保最小值不小于 0（防止无交集时出现负数）
    w = np.maximum(xright - xleft, 0)  # 交集区域的宽度
    h = np.maximum(yright - yleft, 0)  # 交集区域的高度

    # 计算交集区域面积
    inter = w * h

    # 计算 bbox1 和 bbox2 各自的面积
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1 的面积
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2 的面积

    # 计算并集面积（Union）：两个矩形的总面积 - 交集面积
    union = area_bbox1 + area_bbox2 - inter

    # 计算 IoU = 交集 / 并集，避免除以 0
    iou = inter / union if union > 0 else 0

    return iou  # 返回 IoU 值


# 测试代码
if __name__ == "__main__":
    # 定义两个矩形框（bbox）
    bbox1 = [0, 0, 2, 2]  # 左上角 (0,0) 到 右下角 (2,2)
    bbox2 = [1, 1, 3, 3]  # 左上角 (1,1) 到 右下角 (3,3)

    # 计算 IoU
    iou = single_iou(bbox1, bbox2)

    # 打印 IoU 结果
    print("IoU 值:", iou)
