#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig
from .modeling_qwen2_flash import Qwen2Model_Flash, Qwen2ForCausalLM_Flash

class LlavaQwenConfig_Flash(Qwen2Config):
    model_type = "llava_qwen_flash"


class LlavaQwenModel_Flash(LlavaMetaModel, Qwen2Model_Flash):
    config_class = LlavaQwenConfig_Flash

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel_Flash, self).__init__(config)


class LlavaQwenForCausalLM_Flash(Qwen2ForCausalLM_Flash, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig_Flash

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM_Flash.__init__(self, config)
        config.model_type = "llava_qwen_flash"
        # config.rope_scaling = None

        self.model = LlavaQwenModel_Flash(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # print("inputs_embeds.shape:", inputs_embeds.shape)
        if dpo_forward:
            outputs, labels = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels
                
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            self.model.image_token_posi = [-1]     
            self.model.prompt_len = None
            self.model.image_tokens = [0]
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None,
                                       cache_position=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            cache_position=cache_position, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

    @torch.no_grad()
    def chat(
        self,
        video_path,
        tokenizer,
        question,
        chat_history=None,
        return_history=False,
        max_num_frames=512,
        media_dict=None,
        generation_config=None,
    ):
        """
        视频推理接口，供 lmms_eval 评估框架调用。

        Args:
            video_path:         视频文件路径
            tokenizer:          Qwen2 tokenizer
            question:           问题文本（不含 <image> 占位符）
            chat_history:       历史对话（暂不使用，预留接口）
            return_history:     是否返回更新后的 chat_history
            max_num_frames:     最多采样帧数（对齐到 mm_local_num_frames 的倍数）
            media_dict:         视频读取参数，支持 video_read_type / start / end
            generation_config:  dict，直接传给 generate(**generation_config)
        """
        from llava.video_utils import VIDEO_READER_FUNCS
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        if generation_config is None:
            generation_config = {}
        if media_dict is None:
            media_dict = {'video_read_type': 'decord'}

        # ── 1. 帧采样数对齐 ───────────────────────────────────────────
        mm_local_num_frames = getattr(self.config, 'mm_local_num_frames', 4)
        num_frames = (max_num_frames // mm_local_num_frames) * mm_local_num_frames
        num_frames = max(num_frames, mm_local_num_frames)

        # ── 2. 读取视频帧 ─────────────────────────────────────────────
        video_read_type = media_dict.get('video_read_type', 'decord')
        clip = None
        if 'start' in media_dict and 'end' in media_dict:
            clip = (media_dict['start'], media_dict['end'])

        read_func = VIDEO_READER_FUNCS[video_read_type]
        frames, _, _, _ = read_func(
            video_path,
            num_frames=num_frames,
            sample='middle',
            max_num_frames=num_frames,
            clip=clip,
            local_num_frames=mm_local_num_frames,
        )
        # frames: numpy (T, H, W, C) uint8

        # ── 3. 帧数向下对齐到 mm_local_num_frames 的倍数 ──────────────
        T = frames.shape[0]
        T = max((T // mm_local_num_frames) * mm_local_num_frames, mm_local_num_frames)
        frames = frames[:T]

        # ── 4. 图像预处理 → (T, C, H, W) ─────────────────────────────
        vision_tower = self.get_vision_tower()
        processor = vision_tower.image_processor
        video_tensor = processor.preprocess(frames, return_tensors='pt')['pixel_values']
        video_tensor = video_tensor.to(dtype=self.dtype, device=self.device)

        # ── 5. 构建 Qwen2 对话 prompt ─────────────────────────────────
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + question
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ── 6. Tokenize（<image> → IMAGE_TOKEN_INDEX）─────────────────
        input_ids = tokenizer_image_token(
            text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # ── 7. 生成 ───────────────────────────────────────────────────
        output_ids = self.generate(
            inputs=input_ids,
            images=[video_tensor],
            modalities=["video"],
            **generation_config,
        )

        # ── 8. 解码 ───────────────────────────────────────────────────
        # generate 内部以 inputs_embeds 模式调用 super().generate()，
        # 输出 output_ids 只含新生成 token；用 skip_special_tokens=True 直接解码。
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if return_history:
            if chat_history is None:
                chat_history = []
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": response})
            return response, chat_history
        return response


AutoConfig.register("llava_qwen_flash", LlavaQwenConfig_Flash)
AutoModelForCausalLM.register(LlavaQwenConfig_Flash, LlavaQwenForCausalLM_Flash)
