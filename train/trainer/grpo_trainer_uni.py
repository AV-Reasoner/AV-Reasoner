import os
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['REGIONAL_POOL'] = '2x'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['SKIP_LOAD_VIT'] = '1'


import av
import torch
from PIL import Image
import numpy as np
import transformers
from typing import Optional
import librosa
import whisper
import copy
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_speech_image_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IMAGE_TOKEN_INDEX
from moviepy.editor import VideoFileClip

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class OLAGRPOUniTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model_id,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        args.gradient_checkpointing_kwargs={"use_reentrant": False}
        args.optimize_cuda_cache = True
        args.save_safetensors=False

        tokenizer, model, image_processor, _ = load_pretrained_model(model_id, None)
        import torch.nn.utils.parametrize as P
        model.get_vision_tower().requires_grad_(False)
        model.get_speech_encoder().requires_grad_(False)
        pad_token_id = tokenizer.pad_token_id
        self.processor = tokenizer
        processing_class = tokenizer
        self.image_processor = image_processor




        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, # **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        self.embedding_model = None

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        _, self.ref_model, _, _ = load_pretrained_model(model_id, None)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def load_audio(self,audio_file_name):
        speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
        if len(speech_wav.shape) > 1:
            speech_wav = speech_wav[:, 0]
        speech_wav = speech_wav.astype(np.float32)
        CHUNK_LIM = 480000
        SAMPLE_RATE = 16000
        speechs = []
        speech_wavs = []

        if len(speech_wav) <= CHUNK_LIM:
            speech = whisper.pad_or_trim(speech_wav)
            speech_wav = whisper.pad_or_trim(speech_wav)
            speechs.append(speech)
            speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
        else:
            for i in range(0, len(speech_wav), CHUNK_LIM):
                chunk = speech_wav[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                speechs.append(chunk)
                speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
        mels = []
        for chunk in speechs:
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(chunk)

        mels = torch.cat(mels, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)
        if mels.shape[0] > 25:
            mels = mels[:25]
            speech_wavs = speech_wavs[:25]

        speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])
        return mels, speech_length, speech_chunks, speech_wavs

    def process_audio(self,file):
        speechs = []
        speech_lengths = []
        speech_wavs = []
        speech_chunks = []
        speech, speech_length, speech_chunk, speech_wav = self.load_audio(file)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.to('cuda'))
        return speechs,speech_lengths,speech_wavs,speech_chunks



    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print(inputs)
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_inputs = {}
        prompts = []
        for x in inputs:
            base_dir_video = ""
            base_dir_audio = ""


            if x["task"] != "dvd_count" and x["task"] != "RepCount": 
                try:
                    speechs,speech_lengths,speech_wavs,speech_chunks = self.process_audio(base_dir_audio+x["audio"])
                except: # extract from video
                    _video = VideoFileClip(base_dir_video+x["video"])
                    audio = _video.audio
                    audio.write_audiofile(base_dir_audio+x["audio"], logger=None)
                    _video.close()
                    speechs,speech_lengths,speech_wavs,speech_chunks = self.process_audio(base_dir_audio+x["audio"])
            else: # no audio
                speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
                speech_lengths = [torch.LongTensor([3000]).to('cuda')]
                speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
                speech_chunks = [torch.LongTensor([1]).to('cuda')]


            if "image" in x and x["image"] is not None:
                image = [Image.open(base_dir_video+x["image"])]
                image_sizes = [image[0].size]
                self.image_processor.do_resize = False
                self.image_processor.do_center_crop = False
                image_tensor, image_highres_tensor = [], []

                for visual in image:
                    image_tensor_, image_highres_tensor_ = process_anyres_highres_image(visual, self.image_processor)
                    image_tensor.append(image_tensor_)
                    image_highres_tensor.append(image_highres_tensor_)
                if all(x.shape == image_tensor[0].shape for x in image_tensor):
                    image_tensor = torch.stack(image_tensor, dim=0)
                if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
                    image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
                if type(image_tensor) is list:
                    image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
                else:
                    image_tensor = image_tensor.bfloat16().to("cuda")
                if type(image_highres_tensor) is list:
                    image_highres_tensor = [_image.bfloat16().to("cuda") for _image in image_highres_tensor]
                else:
                    image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")
                    
                
                qs = x["prompt"]
            
                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                message = [{'role': 'user', 'content': qs}]
                prompts.append(message)
                input_ids = tokenizer_speech_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
                pad_token_ids = 151643

                attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                
                prompt_inputs["input_ids"] = input_ids
                prompt_inputs["modalities"] = ["image"]
                prompt_inputs["attention_mask"] = attention_masks
                prompt_inputs["image_tensor"] = image_tensor
                prompt_inputs["image_highres_tensor"] = image_highres_tensor
                prompt_inputs["image_sizes"] = image_sizes
                prompt_inputs["speechs"] = speechs
                prompt_inputs["speech_lengths"] = speech_lengths
                prompt_inputs["speech_wavs"] = speech_wavs
                prompt_inputs["speech_chunks"] = speech_chunks
                prompt_inputs["stopping_criteria"] = stopping_criteria

            elif "video" in x and x["video"] is not None:
                container = av.open(base_dir_video+x["video"])
                total_frame_num = container.streams.video[0].frames
                fps = round(container.streams.video[0].average_rate)
                frame_idx = np.linspace(0, total_frame_num - 1, 64, dtype=int).tolist()
                frames = []
                for i, frame in enumerate(container.decode(video=0)): 
                    if i in frame_idx:
                        frames.append(frame.to_ndarray(format='rgb24'))

                spare_frames = np.array(frames)
                video = [Image.fromarray(frame) for frame in spare_frames]
                video_processed = []
                for idx, frame in enumerate(video):
                    self.image_processor.do_resize = False
                    self.image_processor.do_center_crop = False
                    frame = process_anyres_video(frame, self.image_processor)

                    if frame_idx is not None and idx in frame_idx:
                        video_processed.append(frame.unsqueeze(0))
                    elif frame_idx is None:
                        video_processed.append(frame.unsqueeze(0))

                if frame_idx is None:
                    frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()

                video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
                
                qs = x["prompt"]
            
                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                message = [{'role': 'user', 'content': qs}]
                prompts.append(message)
                input_ids = tokenizer_speech_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
                pad_token_ids = 151643

                attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

                prompt_inputs["input_ids"] = input_ids
                prompt_inputs["modalities"] = ["video"]
                prompt_inputs["attention_mask"] = attention_masks
                prompt_inputs["image_tensor"] = video_processed
                prompt_inputs["image_highres_tensor"] = video_processed
                prompt_inputs["image_sizes"] = None
                prompt_inputs["speechs"] = speechs
                prompt_inputs["speech_lengths"] = speech_lengths
                prompt_inputs["speech_wavs"] = speech_wavs
                prompt_inputs["speech_chunks"] = speech_chunks
                prompt_inputs["stopping_criteria"] = stopping_criteria


        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1

            all_completions = []

            for i in range(num_generations):  # -1 because we already have one generation
                completion = unwrapped_model.generate(
                            inputs=prompt_inputs["input_ids"],
                            images=prompt_inputs["image_tensor"],
                            images_highres=prompt_inputs["image_highres_tensor"],
                            image_sizes=prompt_inputs["image_sizes"],
                            modalities=prompt_inputs["modalities"],
                            speech=prompt_inputs["speechs"],
                            speech_lengths=prompt_inputs["speech_lengths"],
                            speech_chunks=prompt_inputs["speech_chunks"],
                            speech_wav=prompt_inputs["speech_wavs"],
                            attention_mask=prompt_inputs["attention_mask"],
                            stopping_criteria=[prompt_inputs["stopping_criteria"]],
                            generation_config=temp_generation_config

                        )
                
                all_completions.append(completion)

            # Stack all completions and pad if needed
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []

            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            # Stack all padded completions
            prompt_completion_ids = torch.cat(padded_completions, dim=0)
            
            # prompt_completion_ids = torch.cat(padded_completions, dim=0)
            completion_ids = torch.cat(padded_completions, dim=0)
            input_ids = prompt_inputs["input_ids"].repeat_interleave(self.num_generations, dim=0)  #prompt_inputs["input_ids"].repeat(1,2)
            prompt_completion_ids = torch.concat([input_ids, completion_ids],dim=-1)
        # import pdb; pdb.set_trace()


        def get_per_token_logps(model, input_ids, completion_length=None, prompt_inputs=None):

            # import pdb; pdb.set_trace()
            logits = []


            for i in range(len(input_ids)):
                logits.append(model(input_ids=input_ids[i:i+1],
                                    images=prompt_inputs["image_tensor"],
                                    images_highres=prompt_inputs["image_highres_tensor"],
                                    image_sizes=prompt_inputs["image_sizes"],
                                    modalities=prompt_inputs["modalities"],
                                    speech=prompt_inputs["speechs"],
                                    speech_lengths=prompt_inputs["speech_lengths"],
                                    speech_chunks=prompt_inputs["speech_chunks"],
                                    speech_wav=prompt_inputs["speech_wavs"],
                            ).logits)  # (B, L, V)
            logits = torch.cat(logits, dim=0)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            logits = logits[:,-completion_length:]
            input_ids = input_ids[:,-completion_length:]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids,
                                              prompt_inputs=prompt_inputs, completion_length=max_length,
                                             )
        # print(per_token_logps.shape)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        # per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, prompt_inputs=prompt_inputs, completion_length=max_length)


        # Compute the KL divergence between the model and the reference model
        diff = ref_per_token_logps - per_token_logps
        diff = torch.clamp(diff, min=-11.0, max=11.0)

        per_token_kl = torch.exp(diff) - (diff) - 1

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # import pdb; pdb.set_trace()
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]): # true
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions,embedding_model = self.embedding_model,**reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl) # default 0.04

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # import pdb; pdb.set_trace()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["advantages"].append(self.accelerator.gather_for_metrics(advantages).mean().item())

        self._metrics["reward_mean"].append(self.accelerator.gather_for_metrics(mean_grouped_rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # import pdb; pdb.set_trace()
        torch.cuda.empty_cache()
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()