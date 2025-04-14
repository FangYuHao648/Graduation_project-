import math
import time
import warnings

import cv2
import numpy as np
import torch
import yaml
from QtFusion.models import Detector, HeatmapGenerator
from QtFusion.path import abs_path

from datasets.Qrcode.label_name import Chinese_name
from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression

auto_device = "0" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)

# Example of setting init_params
init_params = {
    'device': auto_device,
    'conf': 0.25,
    'iou': 0.45,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'max_det': 1000
}


def count_classes(det_info, class_names):
    """
    Count the number of each class in the detection info.

    :param det_info: List of detection info, each item is a list like [class_name, bbox, conf, class_id]
    :param class_names: List of all possible class names
    :return: A list with counts of each class
    """
    count_dict = {name: 0 for name in class_names}  # 创建一个字典，用于存储每个类别的数量
    for info in det_info:  # 遍历检测信息
        class_name = info['class_name']  # 获取类别名称
        if class_name in count_dict:  # 如果类别名称在字典中
            count_dict[class_name] += 1  # 将该类别的数量加1

    # Convert the dictionary to a list in the same order as class_names
    count_list = [count_dict[name] for name in class_names]  # 将字典转换为列表，列表的顺序与class_names相同
    return count_list  # 返回列表


def read_names_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        names = data.get('names', [])
        return names


init_names = read_names_from_yaml(abs_path("data/coco.yaml"))


def model_switch(model):
    """ Model switch to deploy status """
    from yolov6.layers.common import RepVGGBlock
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

    LOGGER.info("Switch model to deploy modality.")


class YOLOv6Detector(Detector):
    def __init__(self, device=auto_device, imgsz=640, params=None):
        super().__init__(device, imgsz)
        self.stride = None
        self.model = None

        self.img0 = None
        self.img = None

        self.device = device

        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.device}' if cuda else 'cpu')

        self.half = False

        self.params = params if params else init_params
        self.conf = self.params.get('conf', 0.25)
        self.iou = self.params.get('iou', 0.45)
        self.classes = self.params.get('classes', None)
        self.agnostic_nms = self.params.get('agnostic_nms', False)
        self.max_det = self.params.get('max_det', 1000)
        self.names = list(Chinese_name.values())  # 获取所有类别的中文名称

        # 创建heatmap
        self.heatmap = HeatmapGenerator(heatmap_intensity=0.4, hist_eq_threshold=200)

    def load_model(self, model_path):
        self.model = DetectBackend(model_path, device=self.device)
        self.model.names = self.names if "best" in model_path else init_names
        self.stride = self.model.stride
        self.imgsz = self.check_img_size(self.imgsz, s=self.stride)  # 检查图像尺寸

        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        model_switch(self.model.model)

        if isinstance(self.model.names, dict):
            names_dict = self.model.names
            self.names = [Chinese_name[v] if v in Chinese_name else v for v in names_dict.values()]
        elif isinstance(self.model.names, list):
            self.names = [Chinese_name[v] if v in Chinese_name else v for v in self.model.names]
        else:
            raise TypeError("Unsupported type for self.model.names")

        detect_module = list(self.model.model.children())[-2]
        desired_layer = list(detect_module.children())[-2]
        self.heatmap.register_hook(reg_layer=desired_layer)
        self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).
                   type_as(next(self.model.model.parameters())))  # 预热

    def preprocess(self, img):
        self.img0 = img
        self.img = letterbox(img, self.imgsz, stride=self.stride)[0]
        # Convert
        self.img = self.img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        self.img = torch.from_numpy(np.ascontiguousarray(self.img)).to(self.device)

        self.img = self.img.half() if self.half else self.img.float()  # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)

        return self.img

    def predict(self, img):
        with torch.no_grad():
            pred_results = self.model(img)
            img = cv2.cvtColor(self.img0, cv2.COLOR_BGR2RGB)
            superimposed_img = self.heatmap.get_heatmap(img)
        return pred_results, superimposed_img

    def set_param(self, params):
        self.params.update(params)
        self.conf = self.params.get('conf', 0.25)
        self.iou = self.params.get('iou', 0.45)
        self.classes = self.params.get('classes', None)
        self.agnostic_nms = self.params.get('agnostic_nms', False)
        self.max_det = self.params.get('max_det', 1000)

    def postprocess(self, prediction):
        det = non_max_suppression(prediction, self.conf, self.iou, self.classes, self.agnostic_nms,
                                  max_det=self.max_det)[0]
        det_results = []
        if len(det):
            # Rescale boxes from img_size to original image size
            det[:, :4] = self.rescale(self.img.shape[2:], det[:, :4], self.img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                bbox = [round(x.item(), 2) for x in xyxy[0: 4]]
                class_id = int(cls)
                class_name = Chinese_name[self.names[class_id]] if self.names[class_id] in Chinese_name else \
                    self.names[class_id]
                result = {
                    "class_name": class_name,
                    "bbox": bbox,
                    "score": conf.item(),
                    "class_id": class_id,
                }
                det_results.append(result)
        return det_results

    def check_img_size(self, img_size, s=32, floor=0):
        if isinstance(img_size, int):  # 整数，比如img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # 列表，比如img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"不支持的img_size类型: {type(img_size)}")

        if new_size != img_size:
            print(f'警告: --img-size {img_size} 必须是最大步长 {s} 的倍数, 更新为 {new_size}')
        return new_size if isinstance(img_size, list) else [new_size] * 2

    @staticmethod
    def make_divisible(x, divisor):
        # 确保x是divisor的倍数
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        """Rescale the output to the original image shape"""
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes


if __name__ == '__main__':
    # 创建检测器实例
    detector = YOLOv6Detector()

    # 加载模型
    start_time = time.time()
    detector.load_model(abs_path("weights/best_ckpt.pt"))
    end_time = time.time()
    print(f"模型加载耗时：{end_time - start_time:.3f}秒")

    # 读取并预处理图像
    start_time = time.time()
    img = cv2.imread(abs_path("test_media/Qrcode_SIXU_A00076.jpg"))
    img = detector.preprocess(img)
    end_time = time.time()
    print(f"图像预处理耗时：{end_time - start_time:.3f}秒")

    # 进行预测
    start_time = time.time()
    pred, _ = detector.predict(img)
    end_time = time.time()
    print(f"模型预测耗时：{end_time - start_time:.3f}秒")

    # 后处理
    start_time = time.time()
    results = detector.postprocess(pred)
    end_time = time.time()
    print(f"后处理耗时：{end_time - start_time:.3f}秒")

    # 打印结果
    print(results)
