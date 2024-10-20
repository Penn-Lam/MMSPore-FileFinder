# 预处理图片和视频，建立索引，加快搜索速度

# 导入并发执行器模块，用于实现多线程或多进程任务
import concurrent.futures
# 导入日志模块，用于记录程序运行过程中的信息
import logging
# 导入错误跟踪模块，用于获取和打印异常堆栈信息
import traceback

# 导入MindSpore库
import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor

# 导入OpenCV库，用于图像与视频处理
import cv2
# 导入NumPy库，用于科学计算，特别是数组操作
import numpy as np
# 导入requests库，用于发送HTTP请求
import requests
# 导入Pillow库中的Image类，用于处理图像文件
from PIL import Image
# 导入进度条库，用于显示循环或迭代过程中的进度
from tqdm import trange
# 导入Transformers库中的AutoModelForZeroShotImageClassification类，用于零样本图像分类任务
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
from clip import clip
from transformers import CLIPProcessor, CLIPModel

from config import *

logger = logging.getLogger(__name__)

def load_clip_model():
    # 检查是否存在转换后的 MindSpore 模型
    if os.path.exists(MODEL_WEIGHT_PATH):
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        param_dict = ms.load_checkpoint(MODEL_WEIGHT_PATH)
        ms.load_param_into_net(model, param_dict)
        logger.info(f"Loaded {len(param_dict)} parameters from {MODEL_WEIGHT_PATH}")
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    else:
        # 如果不存在，则从 PyTorch 模型转换
        pt_path = "./models/ViT-B-32.pt"
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"PyTorch model file not found: {pt_path}")
        
        from convert_model import convert_pytorch_to_mindspore
        convert_pytorch_to_mindspore(pt_path, MODEL_WEIGHT_PATH)
        
        model, _ = clip.load("ViT-B/32", device=DEVICE)
        param_dict = ms.load_checkpoint(MODEL_WEIGHT_PATH)
        ms.load_param_into_net(model, param_dict)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    return model, processor

model, processor = load_clip_model()

# 修改设备配置
ms.set_context(device_target=DEVICE)

logger.info("Model loaded.")


def get_image_feature(images):
    feature = None
    try:
        if isinstance(images, list):
            inputs = processor(images=images, return_tensors="np", padding=True)['pixel_values']
        else:
            inputs = processor(images=images, return_tensors="np")['pixel_values']
        inputs = Tensor(inputs, dtype=ms.float32)
        feature = model.encode_image(inputs)
        return feature.asnumpy()  # 确保返回numpy数组
    except Exception as e:
        logger.warning(f"处理图片报错：{repr(e)}")
        traceback.print_stack()
    return feature


def get_image_data(path: str, ignore_small_images: bool = True):
    """
    获取图片像素数据，如果出错返回 None
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片数据，如果出错返回 None
    """
    try:
        image = Image.open(path)
        if ignore_small_images:
            width, height = image.size
            if width < IMAGE_MIN_WIDTH or height < IMAGE_MIN_HEIGHT:
                return None
        image = image.convert('RGB')
        image = np.array(image)
        return image
    except Exception as e:
        logger.warning(f"打开图片报错：{path} {repr(e)}")
        return None


def process_image(path, ignore_small_images=True):
    """
    处理图片，返回图片特征
    :param path: string, 图片路径
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    image = get_image_data(path, ignore_small_images)
    if image is None:
        return None
    feature = get_image_feature(image)
    return feature  # 现在返回的是numpy数组


def process_images(path_list, ignore_small_images=True):
    """
    处理图片，回图片特征
    :param path_list: string, 图片路径列表
    :param ignore_small_images: bool, 是否忽略尺寸过小的图片
    :return: <class 'numpy.nparray'>, 图片特征
    """
    images = []
    for path in path_list.copy():
        image = get_image_data(path, ignore_small_images)
        if image is None:
            path_list.remove(path)
            continue
        images.append(image)
    if not images:
        return None, None
    feature = get_image_feature(images)
    return path_list, feature  # 现在返回的feature是numpy数组


def process_web_image(url):
    """
    处理网络图片，返回图片特征
    :param url: string, 图片URL
    :return: <class 'numpy.nparray'>, 图片特征
    """
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        logger.warning("获取图片报错：%s %s" % (url, repr(e)))
        return None
    feature = get_image_feature(image)
    return feature


def get_frames(video: cv2.VideoCapture):
    """ 
    获取视频的帧数据
    :return: (list[int], list[array]) (帧编号列表, 帧像素数据列表) 元组
    """
    frame_rate = round(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"fps: {frame_rate} total: {total_frames}")
    ids, frames = [], []
    for current_frame in trange(
            0, total_frames, FRAME_INTERVAL * frame_rate, desc="当前进度", unit="frame"
    ):
        ret, frame = video.read()
        if not ret:
            break
        ids.append(current_frame // frame_rate)
        frames.append(frame)
        if len(frames) == SCAN_PROCESS_BATCH_SIZE:
            yield ids, frames
            ids = []
            frames = []
        for _ in range(FRAME_INTERVAL * frame_rate - 1):
            video.grab()  # 跳帧
    yield ids, frames


def process_video(path):
    """
    处理视频并返回处理完成的数据
    返回一个生成器，每调用一次则返回视频下一个帧的数据
    :param path: string, 视频路径
    :return: [int, <class 'numpy.nparray'>], [当前是第几帧（被集的才算），图片特征]
    """
    logger.info(f"处理视频中：{path}")
    try:
        video = cv2.VideoCapture(path)
        for ids, frames in get_frames(video):
            features = get_image_feature(frames)
            if features is None:
                logger.warning("features is None")
                continue
            for id, feature in zip(ids, features):
                yield id, feature
    except Exception as e:
        logger.warning(f"处理视频出错：{path} {repr(e)}")
        return


def process_text(input_text):
    if not input_text:
        logger.warning("输入文本为空")
        return None
    try:
        # 使用 tokenize 方法处理文本
        text = clip.tokenize([input_text])
        # 直接使用 model 的 encode_text 方法
        text_features = model.encode_text(text)
        return text_features.asnumpy().squeeze()  # 移除多余的维度
    except Exception as e:
        logger.warning(f"处理文字报错：{repr(e)}")
        traceback.print_stack()
    return None


def match_text_and_image(text_feature, image_feature):
    """
    匹配文字和图片，返回余弦相似度
    :param text_feature: <class 'numpy.nparray'>, 文字特征
    :param image_feature: <class 'numpy.nparray'>, 图片特征
    :return: <class 'numpy.nparray'>, 文字和图片的余弦相似度shape=(1, 1)
    """
    score = (image_feature @ text_feature.T) / (
            np.linalg.norm(image_feature) * np.linalg.norm(text_feature)
    )
    return score


def normalize_features(features):
    """
    归一化
    :param features: [<class 'numpy.nparray'>], 特征
    :return: <class 'numpy.nparray'>, 归一化后的特征
    """
    features_tensor = Tensor(features, dtype=ms.float32)
    norm = ops.norm(features_tensor, dim=-1, keepdim=True)
    normalized_features = features_tensor / norm
    return normalized_features.asnumpy()


def multithread_normalize(features):
    """
    多线程执行归一化，只有对大矩阵效果才好
    :param features:  [<class 'numpy.nparray'>], 特征
    :return: <class 'numpy.nparray'>, 归一化后的特征
    """
    num_threads = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        chunk_size = len(features) // num_threads
        chunks = [
            features[i: i + chunk_size] for i in range(0, len(features), chunk_size)
        ]
        normalized_chunks = executor.map(normalize_features, chunks)
    return np.concatenate(list(normalized_chunks))


def match_batch(
        positive_feature,
        negative_feature,
        image_features,
        positive_threshold,
        negative_threshold,
):
    """
    匹配image_feature列表并返回余弦相似度
    :param positive_feature: <class 'numpy.ndarray'>, 正向提示词特征
    :param negative_feature: <class 'numpy.ndarray'>, 反向提示词特征
    :param image_features: [<class 'numpy.ndarray'>], 图片特征列表
    :param positive_threshold: int/float, 正向提示分数阈值，高于此分数才显示
    :param negative_threshold: int/float, 反向提示分数阈值，低于此分数才显示
    :return: [<class 'numpy.nparray'>], 提示词和每个图片余弦相似度列表，里面每个元素的shape=(1, 1)，如果小于正向提示分数阈值或大于反向提示分数阈值则会置0
    """
    # 确保输入是numpy数组
    image_features = np.array(image_features)
    positive_feature = np.array(positive_feature)
    
    # 归一化特征
    image_features = normalize_features(image_features)
    positive_feature = normalize_features(positive_feature)
    
    # 转换为MindSpore的Tensor
    image_features = Tensor(image_features, dtype=ms.float32)
    positive_feature = Tensor(positive_feature, dtype=ms.float32)
    
    # 计算余弦相似度
    similarity = ops.matmul(image_features, positive_feature.T).squeeze()
    
    # 应用阈值
    scores = ops.where(similarity < Tensor(positive_threshold / 100, dtype=ms.float32), 
                       Tensor(0, dtype=ms.float32), 
                       similarity)
    
    if negative_feature is not None:
        negative_feature = normalize_features(np.array(negative_feature))
        negative_feature = Tensor(negative_feature, dtype=ms.float32)
        negative_similarity = ops.matmul(image_features, negative_feature.T).squeeze()
        scores = ops.where(negative_similarity > Tensor(negative_threshold / 100, dtype=ms.float32), 
                           Tensor(0, dtype=ms.float32), 
                           scores)
    
    return scores.asnumpy()
