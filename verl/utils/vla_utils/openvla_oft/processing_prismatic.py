"""
processing_prismatic.py

面向 Prismatic VLM 的 HuggingFace 风格预处理器定义，继承自 `ProcessorMixin`。默认配置
为 `siglip-224px+7b`。
"""

from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === 图像处理 ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """给定一张 PIL.Image，通过在高度/宽度周围添加对称边框将其填充为正方形。"""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        """
        初始化一个 PrismaticImageProcessor，作为 torchvision transform 的包装器；该 transform 将由
        TIMM 创建，并经过修改以遵循我们自定义的 `image_resize_strategy` 逻辑。
        @param use_fused_vision_backbone: 布尔值，指示单视觉骨干还是融合（双）视觉骨干
        @param image_resize_strategy: Prismatic 图像缩放策略，取值 < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] 输入图像尺寸，元组 (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] 插值方式，字符串（默认："bicubic"）
        @param mean: [TIMM :: `data_cfg`] 归一化均值，浮点元组（若为 `fused_backbone` 则为二元组）
        @param std: [TIMM :: `data_cfg`] 归一化标准差，浮点元组（若为 `fused_backbone` 则为二元组）
        """
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # 处理 `None` 默认值
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        # TIMM `data_cfg` 参数
        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds

        # 通过 TIMM 获取 torchvision transforms =>> 需要解析出特定的 "functional" transform 参数！
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,  # 设为 1.0 以忽略裁剪（初始 Resize 已设置 `input_size`）
                crop_mode="center",  # 默认裁剪模式 —— 当 `crop_pct == 1.0` 时为空操作
                is_training=False,  # 加载 transform 时不做图像增广！
            )

            # [校验] 确保 transform 结构与预期尺寸正确
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
                and (transform.transforms[0].size == self.input_sizes[idx][-1])
                and (transform.transforms[1].size == self.input_sizes[idx][-2:])
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

            # HF 图像处理器必须可 JSON 序列化；因此不能将 torchvision. 作为属性。
            #   => 我们将解析该 transform，并调用 "torchvision.transforms.functional"（`tvf`）。
            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            # 处理 Prismatic `image_resize_strategy`
            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # 将 **kwargs 传给 super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """应用 TIMM Transform 的 `functional` 版本 = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        # [约定] 融合骨干期望"通道堆叠"的输入；我们会在模型侧解包！
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        # [约定] `imgs_t` 是形状为 [3, input_size, input_size] 的张量列表；沿 dim = 0 堆叠
        img_t = torch.vstack(imgs_t)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        预处理一张（或一批）图像；注意，与 `transformers :: BaseImageProcessor` 不同，为简单起见我们
        显式地只处理 PIL.Image.Image 实例。
        @param images: 要预处理的（一批）PIL.Image.Image 实例。
        @param return_tensors: BatchFeature 默认的 Tensor 格式（例如 torch 用 "pt"）；若为 None，则返回 np.ndarray
        @return: 一个 `transformers :: BatchFeature` 实例，仅包含单个键 "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # 对每张图像应用 `self.img_transform`（将返回 torch.Tensor 列表）；堆叠为"批次化"的 Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # 返回 BatchFeature =>> 注意，为了兼容性，构造函数期望 Dict[str, np.ndarray]，因此我们进行转换
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> 同时封装 ImageProcessor 和 Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        为 Prismatic VLM 预处理给定的（一批）文本/图像；将文本转发给底层 LLM 的 tokenizer，
        将图像转发给 PrismaticImageProcessor。
        @param text: 要编码的（一批）文本；必须是字符串或字符串列表。
        @param images: 要预处理的（一批）PIL.Image.Image 实例。
        @param padding: 序列填充策略（若指定多个）< True = "longest" | "max_length" | False >
        @param truncation: 输出序列的截断策略；需要指定 `max_length`
        @param max_length: 截断的最大长度（以 token 为单位）
        @param return_tensors: 返回张量的类型（通常为 "pt" 或 TensorType.PYTORCH）
        @return: 包含 `input_ids`、`attention_mask` 和 `pixel_values` 键的 BatchFeature。
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [校验] 图像与文本输入的数量必须相同！
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer 转发工具 =>> 文档参见 `PreTrainedTokenizerBase` ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
