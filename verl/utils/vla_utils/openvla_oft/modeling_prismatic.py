"""
modeling_prismatic.py

核心的 HuggingFace 风格 PrismaticPreTrainedModel 与 PrismaticForConditionalGeneration 类定义。
继承自默认的 `transformers.PretrainedModel`。目标是独立且自包含，
但完全复刻 `prismatic.models.vlms.prismatic.py` 中的逻辑。
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
    load_component_state_dict,
    find_checkpoint_file,
)
from .constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# 获取 logger
logger = logging.getLogger(__name__)


# === 用于 Monkey-Patching 的工具函数 ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers 会覆盖名称中包含 `gamma` 的参数；我们将为 VisionBackbone.LayerScale 打补丁。
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class ProprioProjector(nn.Module):
    """
    将本体（proprio）状态输入投影到 LLM 的嵌入空间。
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

# === Prismatic 视觉骨干（nn.Module）定义（支持融合骨干） ===
class PrismaticVisionBackbone(nn.Module):
    """
    Prismatic 模型的视觉骨干，负责图像特征提取。

    同时支持单骨干（如 SigLIP）和融合骨干（如 SigLIP + DINOv2）两种配置。
    对于融合骨干，两个模型的特征会沿特征维度拼接。
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        初始化视觉骨干。

        参数：
            use_fused_vision_backbone: 是否使用两个骨干并融合它们的特征
            image_sizes: 每个骨干的图像尺寸列表
            timm_model_ids: 每个骨干使用的 TIMM 模型 ID 列表
            timm_override_act_layers: 每个骨干的激活层覆盖列表
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # 默认值，后续可被覆盖

        # 校验（融合）视觉骨干的数量
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # 创建主特征提取器
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # 若使用融合骨干，则创建次特征提取器
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # 为 HF 兼容性给 LayerScale 模块打补丁
        self._patch_layer_scales()

    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        创建一个具有相应配置的基于 TIMM 的特征提取器模型。

        参数：
            model_id: 要加载的 TIMM 模型 ID
            img_size: 模型的输入图像尺寸
            act_layer: 激活层类型的覆盖设置

        返回：
            配置好的特征提取器模型
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch forward 函数，以提取倒数第二层的特征
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer

    def _patch_layer_scales(self) -> None:
        """
        为所有 LayerScale 模块打补丁，使其与 HF 的参数命名兼容。

        HF Transformers 会覆盖名称中包含 'gamma' 的参数，
        因此我们需要重命名并修改 forward 方法。
        """
        # 为主特征提取器打补丁
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # 若次特征提取器存在，则为其打补丁
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """
        返回视觉骨干输出的视觉 patch 数量。

        返回：
            每张图像的 patch 数量
        """
        return self.featurizer.patch_embed.num_patches

    def get_num_images_in_input(self) -> int:
        """
        返回视觉骨干的输入图像数量。

        返回：
            输入中期望的图像数量
        """
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        设置视觉骨干的输入图像数量。

        参数：
            num_images_in_input: 输入中期望的图像数量
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        实现视觉骨干的前向传播。

        若 `self.use_fused_vision_backbone == True`，则同时使用 SigLIP 和 DINOv2 transformer 提取视觉特征
        （否则仅使用 SigLIP）。支持多图输入（但仅限融合视觉骨干）。

        参数：
            pixel_values (torch.Tensor): 输入图像的像素值，(B, C, H, W)。
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # 拆分 `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> 提取特征 =>> 通道堆叠
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # 将 `pixel_values` 拆分为单张图像（每张 6 个通道：3 个给 SigLIP + 3 个给 DINOv2）
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # 处理每张图像并收集 patch
            all_patches = []
            for img in images:
                # 将每张图像进一步拆分为两组通道（每组 3 个通道）
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # 从 SigLIP 和 DINOv2 两个 vision transformer 获取 patch
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # 沿隐藏维度拼接 SigLIP 与 DINOv2 的 patch
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # 沿 patch 维度拼接所有 patch
            return torch.cat(all_patches, dim=1)


# === Prismatic 投影层（nn.Module）定义 ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # 根据 `use_fused_vision_backbone` 分支 =>> 使用略微不同的 MLP 与投影因子！
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === HF 主类定义 ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Prismatic 因果（视觉条件）语言模型输出的基类；同时暴露视觉特征。"""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # 为 VLM 新增的字段
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # 重要 :: 这个 HF 移植版本 *并非* 用于从零训练；仅用于推理和微调！
        #   => 因此，这段 init_weights 代码并不正确；若要从零训练 VLM，请使用主代码库
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """检查 LLM 是否支持 SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [校验] 对 `config` 字段与依赖版本进行轻量校验
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # 实例化 PrismaticVisionBackbone（可能带融合骨干）
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # 创建多模态投影层
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=config.text_config.hidden_size,
                proprio_dim=config.proprio_dim
            )
        
        # 实例化 LLM 骨干
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size

        # HF 惯例代码 =>> 通过 `_init_weights()` 初始化权重，并设置梯度检查点
        self.post_init()

    # === `PreTrainedModel` 惯例代码 ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # 注意：`Llama-2` 和 `Mistral` 不绑定权重（空操作）

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # 更新 config/实例变量
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        使用向量化操作，将 input_embeddings 中 all_actions_mask 为 True 位置的嵌入
        替换为 noisy_action_features 中对应的嵌入。

        参数：
            input_embeddings: 形状为 (B, S, D) 的张量
            all_actions_mask: 形状为 (B, S) 的布尔张量
            noisy_action_features: 形状为 (B, K, D) 的张量，其中 K 为每个样本中掩码为 True 的数量

        返回：
            修改后的 input_embeddings 张量
        """
        # 克隆输入，避免修改原张量
        new_input_embeddings = input_embeddings.clone()

        # 创建一个与 input_embeddings 形状相同的张量，用于存放加噪动作特征
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # 创建用于拼接的批次索引
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # 获取每个样本中掩码为 True 的索引
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # 将加噪动作特征移动到正确的位置
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features

        # 使用掩码合并原始输入嵌入与加噪动作嵌入
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """从 labels 中获取动作掩码的辅助函数"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """处理视觉特征，可选 FiLM 条件化"""
        if use_film:
            # FiLM：将语言输入融入视觉特征
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # 将 patch 嵌入投影到语言嵌入空间
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """处理本体感知特征并附加到视觉特征之后"""
        if proprio_projector is not None and proprio is not None:
            # projected_patch_embeddings: (bsz, num_patches * num_images, llm_dim)
            # proprio: (bsz, proprio_dim) or (propro_dim,)
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # 为简单起见，直接把本体（proprio）token 附加到投影后的视觉 patch token 末尾
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """构建多模态嵌入与注意力掩码"""
        # 更新注意力掩码
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # 构建多模态嵌入与注意力掩码；将嵌入插入到 <BOS> token 之后（1:）
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        return multimodal_embeddings, multimodal_attention_mask

    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """构建多模态标签，patch 嵌入位置填入 IGNORE_INDEX"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === 核心 Prismatic VLM `forward()` 逻辑 ===
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     output_projector_features: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     proprio=None,
    #     proprio_projector=None,
    #     noisy_actions=None,
    #     noisy_action_projector=None,
    #     diffusion_timestep_embeddings=None,
    #     use_film: bool = False,
    # ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
    #     """在 VLM 中运行一次前向传播，返回一个 PrismaticCausalLMOutputWithPast 实例。"""
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     output_projector_features = output_projector_features if output_projector_features is not None else False
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # 仅在非训练状态下遵循 `use_cache`（即使 `gradient_checkpointing` 已关闭）
    #     use_cache = use_cache and not self.training

    #     # 实例化投影特征的占位符
    #     projected_patch_embeddings = None

    #     # === 使用缓存生成（`input_ids.shape[1] == 1`）=>> 需要 `past_keys_values` ===
    #     if input_ids.shape[1] == 1:
    #         assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
    #         assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
    #         assert labels is None, "Unexpected key `labels` provided during cached generation!"

    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=None,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=None,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === 处理单模态前向传播 ===
    #     elif pixel_values is None:
    #         assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
    #         assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=None,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === 处理多模态前向传播 ===
    #     elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
    #         assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"
            
    #         #test
    #         
    #         #test end
                
    #         # 获取输入嵌入（来自语言模型嵌入）
    #         input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

    #         # 提取动作掩码
    #         all_actions_mask = self._process_action_masks(labels)

    #         # 提取输入嵌入中的语言部分（即去掉动作 token 部分）
    #         language_embeddings = input_embeddings[~all_actions_mask].reshape(
    #             input_embeddings.shape[0], -1, input_embeddings.shape[2]
    #         )  # (B, lang_seq_len, llm_dim)

    #         # 获取视觉特征
    #         projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

    #         # 若提供了本体感知状态则添加
    #         projected_patch_embeddings = self._process_proprio_features(
    #             projected_patch_embeddings, proprio, proprio_projector
    #         )

    #         # [Diffusion] 若提供了扩散时间步嵌入则添加
    #         if diffusion_timestep_embeddings is not None:
    #             # 为简单起见，直接把扩散时间步嵌入附加到投影后的视觉 patch token 末尾
    #             projected_patch_embeddings = torch.cat(
    #                 (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
    #             )

    #         # 处理动作嵌入
    #         if noisy_actions is not None:
    #             # 获取对应所有动作 token 的掩码
    #             all_actions_mask = self._process_action_masks(labels)

    #             # 将加噪动作重整形成单个动作 token
    #             # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
    #             B = noisy_actions.shape[0]
    #             noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

    #             # 将加噪动作 token 投影到语言模型嵌入空间
    #             noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

    #             # 用加噪动作嵌入替换动作 token 的嵌入
    #             input_embeddings = self._replace_input_embeddings(
    #                 input_embeddings, all_actions_mask, noisy_action_features
    #             )
    #         else:
    #             # 将动作 token 的嵌入替换为零
    #             # （随后，位置嵌入会被加到它们上面）
    #             all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    #             input_embeddings = input_embeddings * ~all_actions_mask

    #         # 构建多模态嵌入与注意力掩码
    #         multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
    #             input_embeddings, projected_patch_embeddings, attention_mask
    #         )

    #         # 如有需要，为多模态序列构建标签
    #         multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

    #         # 分发给语言模型
    #         language_model_output = self.language_model(
    #             input_ids=None,
    #             attention_mask=multimodal_attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=multimodal_embeddings,
    #             labels=multimodal_labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )

    #     # === 否则 =>> 视为无效输入！ ===
    #     elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
    #         raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

    #     else:
    #         raise ValueError(
    #             "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
    #             f"=> `input_ids` = {input_ids is not None}\n"
    #             f"=> `attention_mask` = {attention_mask is not None}\n"
    #             f"=> `pixel_values` = {pixel_values is not None}\n"
    #             f"=> `labels` = {labels is not None}\n"
    #             f"=> `input_embeds` = {inputs_embeds is not None}\n"
    #             f"=> `past_key_values` = {past_key_values is not None}\n"
    #             f"=> `use_cache` = {use_cache}"
    #         )

    #     # 解包 `language_model_output` 并返回 PrismaticCausalLMOutputWithPast（若非 `return_dict` 则返回元组）
    #     if not return_dict:
    #         if output_projector_features and (projected_patch_embeddings is not None):
    #             return *language_model_output, projected_patch_embeddings

    #         return language_model_output

    #     return PrismaticCausalLMOutputWithPast(
    #         loss=language_model_output.loss,
    #         logits=language_model_output.logits,
    #         past_key_values=language_model_output.past_key_values,
    #         hidden_states=language_model_output.hidden_states,
    #         attentions=language_model_output.attentions,
    #         projector_features=projected_patch_embeddings,
    #     )

    # === GenerationMixin 方法 ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """借自 `LlamaForCausalLM` 并针对 batch size = 1 做了简化；与原始 PrismaticVLM 逻辑一致。"""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # 处理 `past_key_values`（缓存）=>> 假定 `input_ids` 仅包含未处理的 token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 若传入了 `input_embeds`，我们只想在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 确保在 `model_inputs` 中保留 `pixel_values`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # 委托给语言模型处理（各模型实现方式不同，返回类型也不同）
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)
    
    def _prepare_input_for_action_prediction_verl(self, input_ids, attention_mask):
        """通过添加必要的 token 为动作预测准备输入"""
        # 向 input_ids 添加 (ACTION_DIM * NUM_ACTIONS_CHUNK) 个占位 token，以模拟动作 token
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 向序列添加停止 token（非因果的双向自注意力中需要它，因为它在训练时会出现）
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # 扩展注意力掩码以匹配输入的新形状
        # 注意：目前仅支持 batch size == 1
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction_verl(self, labels, input_ids):
        """若未提供，则创建用于动作预测的 labels 张量"""
        # 用伪造的动作标签扩展 labels 张量
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # 将最后一个标签 token 替换为停止 token
        labels[:, -1] = STOP_INDEX

        return labels
    
    def _verl_discrete_compute_logits(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):#contintue!!!!!
        """执行基于 L1 回归的连续动作预测或离散动作 token 预测。"""
        # 将动作 token 嵌入置零
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # 构建多模态嵌入与注意力掩码
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # 前向传播通过语言模型
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # 提取动作 token 的隐藏状态
        #last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        # actions_hidden_states = last_hidden_states[
        #     :,
        #     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        #     :,
        # ]  # (B, act_chunk_len, D)

        # 处理不同的预测方法
        # if action_head is not None:
        #     # L1 回归预测
        #     normalized_actions = action_head.predict_action(actions_hidden_states)
        #     normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        #     normalized_actions = normalized_actions.float().cpu().detach().numpy()
        # else:
        # 基于离散 token 的预测
      
        compute_logits = language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
            
        return  compute_logits
    
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     unnorm_key: Optional[str] = None,
    #     proprio=None,
    #     proprio_projector=None,
    #     action_head=None,
    #     noisy_action_projector=None,
    #     use_film: bool = False,
    #     **kwargs: str,
    # ) :
    #     """从输入序列预测动作，支持不同的预测方法。

    #     参数：
    #         input_ids: 输入 token id
    #         unnorm_key: 反归一化统计量的键
    #         proprio: 本体感知特征
    #         proprio_projector: 本体感知特征的投影器
    #         action_head: 可选的头，用于 L1 回归或基于扩散的预测
    #         noisy_action_projector: 基于扩散的预测中加噪动作的投影器
    #         use_film: 是否使用 FiLM 条件化
    #         **kwargs: 其他参数，包括 pixel_values 和 attention_mask

    #     返回：
    #         (unnormalized_actions, action_hidden_states) 元组
    #     """
    #     # 如果特殊的空 token（''）尚未出现在 prompt 中冒号（':'）token 之后
    #     # （即 "OUT:" 或 "ASSISTANT:" 之后），则插入它以匹配训练时看到的输入
    #     # if not torch.all(input_ids[:, -1] == 29871):
    #     #     input_ids = torch.cat(
    #     #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    #     #     )
    #     #print("!!!!!!!!!!!!!!Entering forward!!!!!!!!!!")
    #     pixel_values = kwargs["pixel_values"]
    #     attention_mask = kwargs["attention_mask"]
        
    #     # 创建伪造的 labels 张量（动作掩码需要用到）
    #     labels = input_ids.clone()
    #     labels[:] = IGNORE_INDEX

    #     # 获取 prompt 中的 token 数量（不含起始 token）
    #     NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # 减去动作 token 和停止 token

    #     # 通过添加必要的 token 准备输入
    #     #input_ids, attention_mask = self._prepare_input_for_action_prediction_verl(input_ids, attention_mask)
        
    #     #test
    #     placeholder_action_token_ids = (
    #         torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
    #     )
    #     input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

    #     # 向序列添加停止 token（非因果的双向自注意力中需要它，因为它在训练时会出现）
    #     stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
    #     input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

    #     # 扩展注意力掩码以匹配输入的新形状
    #     # 注意：目前仅支持 batch size == 1
    #     mask_extension = (
    #         torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
    #         .to(attention_mask.device)
    #         .to(attention_mask.dtype)
    #     )
    #     attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

    #     #return input_ids, attention_mask
        
    #     #test end
        

    #     # 稍后更新 labels 张量以用于动作掩码计算
    #     #labels = self._prepare_labels_for_action_prediction_verl(labels, input_ids)
    #     #test 
        
    #     ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
    #     labels_extension = (
    #         torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
    #         * ARBITRARY_ACTION_TOKEN_IDX
    #     )
    #     labels = torch.cat([labels, labels_extension], dim=-1)

    #     # 将最后一个标签 token 替换为停止 token
    #     labels[:, -1] = STOP_INDEX

    #     #return labels
        
    #     #test ed
       

    #     # 获取输入嵌入与动作掩码
        
        
        
    #     input_embeddings = self.get_input_embeddings()(input_ids)
        
        
    #     #all_actions_mask = self._process_action_masks(labels)
    #     #test
    #     #current_action_mask = get_current_action_mask(labels)
    #     newline_positions = labels != IGNORE_INDEX

    #     # 计算累加和以识别换行之间的区域
    #     cumsum = torch.cumsum(newline_positions, dim=1)

    #     # 创建掩码
    #     mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    #     # 只提取动作部分
    #     action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    #     current_action_mask = action_tokens_only_mask * mask

    #     #next_actions_mask = get_next_actions_mask(labels)
    #     newline_positions = labels != IGNORE_INDEX

    #     # 计算累加和以识别换行之间的区域
    #     cumsum = torch.cumsum(newline_positions, dim=1)

    #     # 创建掩码
    #     mask = cumsum > ACTION_DIM

    #     # 只提取动作部分
    #     action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
    #     next_actions_mask = action_tokens_only_mask * mask
        
    #     all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        
    #     #test end
        
    #     # 提取语言嵌入
    #     language_embeddings = input_embeddings[~all_actions_mask].reshape(
    #         input_embeddings.shape[0], -1, input_embeddings.shape[2]
    #     )

    #     # 处理视觉特征
    #     #projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
    #     #test
    #     if use_film:
    #         # FiLM：将语言输入融入视觉特征
    #         raise ValueError
    #         patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
    #     else:
    #         patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

    #     projected_patch_embeddings = self.projector(patch_features)
    #     #test end
        
        
    #     # 若提供了本体感知特征则添加
    #     use_proprio = proprio_projector is not None and proprio is not None
    #     if use_proprio:
    #         proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
    #         projected_patch_embeddings = self._process_proprio_features(
    #             projected_patch_embeddings, proprio, proprio_projector
    #         )

    #     # 若提供了扩散则使用扩散，否则使用回归或离散预测
    #     use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

    #     # 计算 patch 数量（若存在，包括本体 token 和/或扩散时间步嵌入）
    #     NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
    #     if use_proprio:
    #         NUM_PATCHES += 1
    #     if use_diffusion:
    #         NUM_PATCHES += 1

    #     if use_diffusion:
    #         raise ValueError
    #         # 采样与输出动作形状相同的随机噪声，作为反向扩散的起始状态
    #         noise = torch.randn(
    #             size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
    #         )

    #         # 执行基于扩散的预测
    #         normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
    #             input_embeddings,
    #             all_actions_mask,
    #             noise,
    #             action_head,
    #             projected_patch_embeddings,
    #             labels,
    #             attention_mask,
    #             NUM_PATCHES,
    #             NUM_PROMPT_TOKENS,
    #             noisy_action_projector,
    #         )
    #     else:
    #         # 执行回归或基于离散 token 的预测
    #         # compute_logits = self._verl_discrete_compute_logits(
    #         #     input_embeddings,
    #         #     all_actions_mask,
    #         #     projected_patch_embeddings,
    #         #     attention_mask,
    #         #     labels,
    #         #     NUM_PATCHES,
    #         #     NUM_PROMPT_TOKENS,
    #         #     action_head,
    #         # )
            
    #         #test
            
    #         all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
    #         input_embeddings = input_embeddings * ~all_actions_mask

    #         # 构建多模态嵌入与注意力掩码
    #         # multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
    #         #     input_embeddings, projected_patch_embeddings, attention_mask
    #         # )
    #         #test
            
    #         projected_patch_attention_mask = None
    #         if attention_mask is not None:
    #             projected_patch_attention_mask = torch.full(
    #                 (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                 fill_value=True,
    #                 dtype=attention_mask.dtype,
    #                 device=attention_mask.device,
    #             )

    #         # 构建多模态嵌入与注意力掩码；在 <BOS> token 之后插入嵌入（1:）
    #         multimodal_embeddings = torch.cat(
    #             [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
    #         )

    #         multimodal_attention_mask = None
    #         if attention_mask is not None:
    #             multimodal_attention_mask = torch.cat(
    #                 [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
    #             )

    #         #return multimodal_embeddings, multimodal_attention_mask
            
    #         #test end

    #         # 通过语言模型前向传播
    #         language_model_output = self.language_model(
    #             input_ids=None,
    #             attention_mask=multimodal_attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=multimodal_embeddings,
    #             labels=None,
    #             use_cache=None,
    #             output_attentions=False,
    #             output_hidden_states=False,
    #             return_dict=True,
    #         )

        
    #         compute_logits = language_model_output.logits[
    #                     :,
    #                     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
    #                 ]
                
    #         #test end

    #     return compute_logits
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values=None,
        attention_mask=None,
        #labels=None,
        proprio=None,
        #proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) :
        """从输入序列预测动作，可选择不同的预测方法。

        参数：
            input_ids: 输入 token id
            unnorm_key: 反归一化统计量的键
            proprio: 本体感知特征
            proprio_projector: 本体感知特征的投影器
            action_head: 可选的头，用于 L1 回归或基于扩散的预测
            noisy_action_projector: 基于扩散的预测中加噪动作的投影器
            use_film: 是否使用 FiLM 条件化
            **kwargs: 额外参数，包括 pixel_values 和 attention_mask

        返回：
            (unnormalized_actions, action_hidden_states) 元组
        """
        # 若提示中冒号（':'）token 之后尚未出现特殊的空 token（''）
        # （在 "OUT:" 或 "ASSISTANT:" 之后），则插入它以匹配训练时见到的输入
        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )
        
        #pixel_values = kwargs["pixel_values"]
        #attention_mask = kwargs["attention_mask"]
        
        # 创建伪 labels 张量（构建动作掩码时需要）
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # # 获取提示词中的 token 数量（不含起始 token）
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # 减去动作 token 与停止 token


        # # 通过添加必要的 token 来准备输入
        # #input_ids, attention_mask = self._prepare_input_for_action_prediction_verl(input_ids, attention_mask)
        
        # #test
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 在序列末尾添加停止 token（非因果的双向自注意力需要它，因为训练时它会出现）
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # 扩展注意力掩码以适配输入的新形状
        # 注意：当前仅支持 batch size == 1
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # # 将最后一个 label token 替换为停止 token
        labels[:, -1] = STOP_INDEX

        
        # 获取输入嵌入与动作掩码
        
        #NUM_PROMPT_TOKENS = kwargs["num_prompt_tokens"]
        
        input_embeddings = self.get_input_embeddings()(input_ids)
        
        
        #all_actions_mask = self._process_action_masks(labels)
        #test
        #current_action_mask = get_current_action_mask(labels)
        newline_positions = labels != IGNORE_INDEX

        # 计算累加和以识别换行之间的区域
        cumsum = torch.cumsum(newline_positions, dim=1)

        # 创建掩码
        mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

        # 只提取动作部分
        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        current_action_mask = action_tokens_only_mask * mask

        #next_actions_mask = get_next_actions_mask(labels)
        newline_positions = labels != IGNORE_INDEX

        # 计算累加和以识别换行之间的区域
        cumsum = torch.cumsum(newline_positions, dim=1)

        # 创建掩码
        mask = cumsum > ACTION_DIM

        # 只提取动作部分
        action_tokens_only_mask = labels > ACTION_TOKEN_BEGIN_IDX
        next_actions_mask = action_tokens_only_mask * mask
        
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        
        #test end
        
        # 提取语言嵌入
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 处理视觉特征
        #projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
        #test
        if use_film:
            # FiLM：将语言输入融入视觉特征
            raise ValueError
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        projected_patch_embeddings = self.projector(patch_features)
        #test end
        
        
        # 若提供了本体感知特征则添加
        use_proprio = self.proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, self.proprio_projector
            )

        # 若提供了扩散则使用扩散，否则使用回归或离散预测
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # 计算 patch 数量（若存在，包括本体 token 和/或扩散时间步嵌入）
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError
            # 采样与输出动作形状相同的随机噪声，作为反向扩散的起始状态
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # 执行基于扩散的预测
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # 执行回归或基于离散 token 的预测
            # compute_logits = self._verl_discrete_compute_logits(
            #     input_embeddings,
            #     all_actions_mask,
            #     projected_patch_embeddings,
            #     attention_mask,
            #     labels,
            #     NUM_PATCHES,
            #     NUM_PROMPT_TOKENS,
            #     action_head,
            # )
            
            #test
            
            all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
            input_embeddings = input_embeddings * ~all_actions_mask

            # 构建多模态嵌入与注意力掩码
            # multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            #     input_embeddings, projected_patch_embeddings, attention_mask
            # )
            #test
            
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # 构建多模态嵌入与注意力掩码；在 <BOS> token 之后插入嵌入（1:）
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )

            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            #return multimodal_embeddings, multimodal_attention_mask
            
            #test end

            # 通过语言模型前向传播
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        
            compute_logits = language_model_output.logits[
                        :,
                        NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                    ]
                
            #test end

        return compute_logits
    
    
  
class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # 计算动作 bin
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 计算反 token 化所需的词表大小 —— 撤销添加的 "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
    
    def load_proprio_projector_weights(self, checkpoint_path_or_repo_id: str):
        """
        为 proprio projector 加载预训练权重。
        
        参数：
            checkpoint_path_or_repo_id: 本地 checkpoint 文件路径或 HF Hub 仓库 ID
        """
        if self.proprio_projector is None:
            raise ValueError("Model was not initialized with use_proprio=True")

        checkpoint_path = find_checkpoint_file(checkpoint_path_or_repo_id, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        self.proprio_projector.load_state_dict(state_dict)

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """通过添加必要的 token 为动作预测准备输入"""
        # 向 input_ids 添加 (ACTION_DIM * NUM_ACTIONS_CHUNK) 个占位 token 以模拟动作 token
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # 在序列末尾添加停止 token（非因果的双向自注意力需要它，因为训练时它会出现）
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # 扩展注意力掩码以适配输入的新形状
        # 注意：当前仅支持 batch size == 1
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """若未提供，则创建用于动作预测的 labels 张量"""
        # 用伪动作标签扩展 labels 张量
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # 将最后一个 label token 替换为停止 token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """使用数据集统计量对动作进行反归一化"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions

    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
    ):
        """执行基于扩散的动作预测"""
        # 设置扩散时间步的值
        action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)
        # 克隆嵌入以便在每个时间步复用
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # 反向扩散：迭代去噪以生成动作预测
        for t in action_head.noise_scheduler.timesteps:
            # 获取扩散模型的噪声预测（以 VLA 潜在嵌入、当前加噪动作
            # 嵌入和扩散时间步嵌入为条件）
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] 用加噪动作替换动作 token 的嵌入
            # （稍后会向其添加位置嵌入）

            # 为简单起见，将扩散时间步嵌入附加到投影后的视觉 token 末尾
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # 将加噪动作重塑并投影到语言嵌入空间
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # 用加噪动作嵌入替换动作 token 嵌入
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # 构建多模态嵌入与注意力掩码
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # 通过语言模型前向传播
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # 提取响应中动作部分的隐藏状态
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # 预测噪声并更新加噪动作：x_t -> x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # 返回最终动作
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
    ):
        """执行基于 L1 回归的连续动作预测或离散动作 token 预测。"""
        # 将动作 token 嵌入置零
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # 构建多模态嵌入与注意力掩码
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # 通过语言模型前向传播
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # 提取动作 token 的隐藏状态
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]  # (B, act_chunk_len, D)

        # 处理不同的预测方法
        if action_head is not None:
            # L1 回归预测
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # 基于离散 token 的预测
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        return normalized_actions, actions_hidden_states
    
    def _verl_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        do_sample=True,
        temperature=1,
    ):
        """执行基于 L1 回归的连续动作预测或离散动作 token 预测。"""
        # 将动作 token 嵌入置零
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # 构建多模态嵌入与注意力掩码
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # 通过语言模型前向传播
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        # 提取动作 token 的隐藏状态
        #last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        # actions_hidden_states = last_hidden_states[
        #     :,
        #     NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        #     :,
        # ]  # (B, act_chunk_len, D)

        # 处理不同的预测方法
        # if action_head is not None:
        #     # L1 regression prediction
        #     normalized_actions = action_head.predict_action(actions_hidden_states)
        #     normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        #     normalized_actions = normalized_actions.float().cpu().detach().numpy()
        # else:
        # 基于离散 token 的预测
        
        #test 
        # NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
        # j = torch.arange(language_model_output.logits.shape[1], device=NUM_PROMPT_TOKENS.device)
        # start = NUM_PROMPT_TOKENS.unsqueeze(1)
        # end = start + ACTION_DIM * NUM_ACTIONS_CHUNK
        # mask_2d = (j >= start) & (j < end)
        # mask = mask_2d.unsqueeze(-1) 
        # actions_masks = mask.expand_as(language_model_output.logits)  
        
        
        NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
        batch_size = language_model_output.logits.shape[0]
        device = language_model_output.logits.device

       
        start_indices = NUM_PROMPT_TOKENS.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]
        #test end
        #test add
        #print("language_model_output",language_model_output.logits.shape[-1])
        #print("self.vocab_size",self.vocab_size) 32000
        #topk_values, topk_indices = torch.topk(language_model_output.logits, k=256, dim=-1)
        #print(topk_indices)
        #assert language_model_output.logits.shape[-1] == self.vocab_size
        #test add
        if do_sample == False:
            #org
            # reponse_ids = language_model_output.logits[
            #         :,
            #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            #     ].argmax(dim=2)
            #reponse_ids = language_model_output.logits[actions_masks].argmax(dim=2)
            #org end
            
            #padding
            # reponse_ids = language_model_output.logits[
            #     torch.arange(batch_size, device=device).unsqueeze(-1),  
            #     seq_indices, 
            #     :
            # ].argmax(dim=2)  
            #padding end
            
            #padding + only get last 256 token
            reponse_ids_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1),  
                seq_indices, 
                :
            ]
            start_index = self.vocab_size - 256 
            response_last256 = reponse_ids_logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            last256_argmax = response_last256.argmax(dim=-1)  # Shape: [batch_size, seq_len]
            reponse_ids = last256_argmax + start_index  # Shape: [batch_size, seq_len]
            #padding + only get last 256 token end
            
            predicted_action_token_ids = reponse_ids.cpu().numpy()
                
        else:
            assert temperature>0
            #org 
            # action_logits  = language_model_output.logits[
            #         :,
            #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            #     ]
            #action_logits = language_model_output.logits[actions_masks]
            #org end
            
            action_logits = language_model_output.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1),  
                seq_indices, 
                :
            ]  
            # padding 
            # scaled_logits = action_logits / temperature
            # probs = torch.softmax(scaled_logits, dim=-1)
            # probs_flat = probs.reshape(-1, probs.shape[-1])  # (B*act_chunk_len, vocab_size)
            # sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)  # (B*act_chunk_len, 1)
            # reponse_ids = sampled_indices_flat.view(action_logits.shape[0], -1)
            # padding end 
            
            #padding + only get last 256 token
            action_logits_last256 = action_logits[..., -256-64:-64]
            scaled_logits = action_logits_last256 / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            assert probs.shape[-1] == 256
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
            original_ids_flat = sampled_indices_flat + (self.vocab_size - 256)
            reponse_ids = original_ids_flat.view(action_logits.shape[0], -1)
            #padding + only get last 256 token end
            
            predicted_action_token_ids = reponse_ids.cpu().numpy()
     
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        #normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)

        return normalized_actions, reponse_ids
        #return normalized_actions, actions_hidden_states

    


    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """从输入序列预测动作，可选择不同的预测方法。

        参数：
            input_ids: 输入 token id
            unnorm_key: 反归一化统计量的键
            proprio: 本体感知特征
            proprio_projector: 本体感知特征的投影器
            action_head: 可选的头，用于 L1 回归或基于扩散的预测
            noisy_action_projector: 基于扩散的预测中加噪动作的投影器
            use_film: 是否使用 FiLM 条件化
            **kwargs: 额外参数，包括 pixel_values 和 attention_mask

        返回：
            (unnormalized_actions, action_hidden_states) 元组
        """
        # 若提示中冒号（':'）token 之后尚未出现特殊的空 token（''）
        # （在 "OUT:" 或 "ASSISTANT:" 之后），则插入它以匹配训练时见到的输入
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # 创建伪 labels 张量（构建动作掩码时需要）
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # 获取提示词中的 token 数量（不含起始 token）
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # 减去动作 token 与停止 token

        # 通过添加必要的 token 来准备输入
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # 更新 labels 张量，供之后计算动作掩码使用
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # 获取输入嵌入与动作掩码
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # 提取语言嵌入
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 处理视觉特征
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # 若提供了本体感知特征则添加
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # 若提供了扩散则使用扩散，否则使用回归或离散预测
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # 计算 patch 数量（若存在，包括本体 token 和/或扩散时间步嵌入）
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # 采样与输出动作形状相同的随机噪声，作为反向扩散的起始状态
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # 执行基于扩散的预测
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # 执行回归或基于离散 token 的预测
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        # 对预测的动作进行反归一化
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states

    def generate_action_verl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        # proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """从输入序列预测动作，可选择不同的预测方法。

        参数：
            input_ids: 输入 token id
            unnorm_key: 反归一化统计量的键
            proprio: 本体感知特征
            proprio_projector: 本体感知特征的投影器
            action_head: 可选的头，用于 L1 回归或基于扩散的预测
            noisy_action_projector: 基于扩散的预测中加噪动作的投影器
            use_film: 是否使用 FiLM 条件化
            **kwargs: 额外参数，包括 pixel_values 和 attention_mask

        返回：
            (unnormalized_actions, action_hidden_states) 元组
        """
        # 若提示中冒号（':'）token 之后尚未出现特殊的空 token（''）
        # （在 "OUT:" 或 "ASSISTANT:" 之后），则插入它以匹配训练时见到的输入
        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]
        do_sample = kwargs["do_sample"]
        temperature = kwargs["temperature"]
        
        # 创建伪 labels 张量（构建动作掩码时需要）
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # 获取提示词中的 token 数量（不含起始 token）
        #NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # 减去动作 token 与停止 token
        #test
        padding_idx = kwargs["padding_idx"]
        num_prompt_tokens = input_ids.ne(padding_idx).sum(dim=1) - 1
        #test end
        

        # 通过添加必要的 token 来准备输入
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # 更新 labels 张量，供之后计算动作掩码使用
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)
        
        # 在此处将 padding 从序列前部移到末尾
        #test
        padding_mask = input_ids.ne(padding_idx)
        assert torch.all(padding_mask==attention_mask.ne(0))
        #print("in predict_action padding_mask:", padding_mask)
        padding_mask = padding_mask.int() 
        sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)
        assert use_film==False
        #test end
        

        # 获取输入嵌入与动作掩码
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # 提取语言嵌入
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 处理视觉特征
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # 若提供了本体感知特征则添加
        use_proprio = self.proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, self.proprio_projector
            )

        # 若提供了扩散则使用扩散，否则使用回归或离散预测
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # 计算 patch 数量（若存在，包括本体 token 和/或扩散时间步嵌入）
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            raise ValueError
            # 采样与输出动作形状相同的随机噪声，作为反向扩散的起始状态
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # 执行基于扩散的预测
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # 执行回归或基于离散 token 的预测
            normalized_actions, reponse_ids = self._verl_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                num_prompt_tokens,
                action_head,
                do_sample=do_sample,
                temperature=temperature,
            )

        # 对预测的动作进行反归一化
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        #verl add!
        actions = actions.reshape(-1 ,NUM_ACTIONS_CHUNK, ACTION_DIM)
        #
        return actions, reponse_ids

    
    
    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """验证并解析用于动作统计量的反归一化键"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """获取策略动作空间的维度。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """获取给定数据集的所有已记录统计量。"""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
