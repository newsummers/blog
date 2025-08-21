# 大语言模型（LLM）微调技术笔记

> Author: **ninehills**  
> Labels: **blog**  
> Created: **2023-05-12T06:49:26Z**  
> Link and comments: <https://github.com/ninehills/blog/issues/92>

> 注：本文较多内容摘抄自文献，并结合开源项目与实践总结而成。

<img width="600" alt="大模型进化树" src="./images/2c708bea-82f9-4554-b098-1f7a320bfd7d.png">

**图 1：大模型进化树**

## 0x00 大模型微调

在预训练后，大模型可以获得解决各种任务的通用能力。但是，越来越多的研究表明，大语言模型的能力可以根据特定目标进一步调整，这就是微调技术。目前主要有两种微调大模型的方法：

1. 指令微调（Instruction Tuning），目标是增强或解锁大语言模型的能力。  
2. 对齐微调（Alignment Fine-tuning），目标是将大语言模型的行为与人类的价值观或偏好对齐。

在 OpenAI 发布的 ChatGPT 中，就主要应用了这些微调技术，取得了显著效果。

<img width="800" alt="InstructGPT 原理" src="./images/f2af11f2-7eee-44d6-b2a0-a4f446bc38cd.png">

**图 2：InstructGPT 原理**

## 0x10 指令微调 (Instruction Tuning)

指令微调本质上是用自然语言格式的实例集合对预训练后的大语言模型进行微调。这种方法与有监督微调及多任务提示训练密切相关。指令微调的关键在于设计格式化的训练示例，使模型能够理解并遵循自然语言指令。

### 0x11 格式化实例构造（微调数据集）

通常，一个指令格式化的实例包括任务描述（instruction）、可选的输入（input）、以及期望输出（output），有时还会包含少量示例作为上下文。

数据集一般由两类方式产出：

1. 格式化已有数据集：将传统 NLP 数据集的内容重新格式化为指令-输入-输出三元组，以用于指令微调。为降低人工格式化成本，可以使用 ChatGPT 等模型自动生成 instruction，例如提示语：“请为这段内容生成一个合理的问题”。  
2. 人工标注数据集：为了获得更好的对齐效果，人工标注是首选，但成本较高。目前很多团队也采用 ChatGPT/GPT-4 生成或辅助生成数据集（如 ShareGPT 的对话历史或由模型生成的问答对）。

数据集可以分为通用任务的数据集和专用领域数据集。通用数据集目前有大量开源资源，专用数据集则针对具体应用场景自行构建。引入多样化的数据来源和任务类型有助于提升模型泛化能力。

目前通用的中文微调数据集示例：

| 数据集                                                                          | 内容                                      |
| ------------------------------------------------------------------------------- | ----------------------------------------- |
| [COIG](https://huggingface.co/datasets/BAAI/COIG)                                | Chinese Open Instruction Generalist project |
| [Stanford Alpaca (Chinese)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)       | Alpaca 数据集中文翻译（ChatGPT 辅助翻译） |
| [BELLE](https://huggingface.co/datasets/BelleGroup/train_2M_CN)                  | BELLE 项目的中文数据集（ChatGPT 生成）    |
| [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)  | Guannaco 模型的对话数据集                 |
| [WebQA(zh)](https://huggingface.co/datasets/suolyer/webqa)                       | 中文网络问答                              |
| [pCLUE](https://github.com/CLUEbenchmark/pCLUE)                                  | 基于提示的大规模预训练数据集，用于多任务学习和零样本学习 |

其余中文数据集与说明可参考以下资源：

- https://github.com/CVI-SZU/Linly/blob/main/instructions/README.md  
- https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/data/README.md

### 0x12 数据集格式示例

典型数据格式为 JSON 三元组：
```json
{"instruction": "", "input": "", "output": ""}
```

若微调已经经过指令微调的模型，应尽量保持数据格式一致以获得最佳效果。下面列举若干常见模型/项目使用的格式示例：

ChatGLM-6B 示例（文本对话风格）：
```
---
Prompt: "编辑文章，使其更吸引读者。..."
Complete: "编辑后文本..."
---
```

Claude 示例（Anthropic 风格）：
```
---
Prompt: "\n\nHuman: Why is the sky blue?\n\nAssistant:"
Complete: "The sky appears blue to us due to how the atmosphere interacts with sunlight. ..."
---
```

Guanaco 示例（指令+输入+响应）：
```
Prompt: """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

Complete: """{output}"""
```

OpenAI ChatML 示例：
```
Prompt: """<|im_start|>system
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09-01
Current date: 2023-03-01<|im_end|>
<|im_start|>user
How are you<|im_end|>
<|im_start|>assistant
I am doing well!<|im_end|>
<|im_start|>user
How are you now?<|im_end|>"
Complete: "{output}"
```

### 0x13 指令微调效果

指令微调常见的收益：

1. 性能改进：指令微调能显著提升模型在指令跟随、问答和任务执行上的性能。研究显示，较小的模型在经过指令微调后，有时能超越未经微调的更大模型。  
2. 任务泛化性：指令微调促使模型理解自然语言指令并在多种任务间迁移，从而提高模型的泛化与灵活应答能力。

### 0x14 对话微调 (Conversation Tuning)

对话微调是指令微调的一个子集，目标是将模型的“补全”能力扩展为长期对话能力。数据格式一般包含对话历史（history）。

ChatGLM-6B 风格对话示例：
```
---
Prompt: """[Round 0]
问：你好，你能帮我解答一个问题吗？
答：当然，请问有什么问题？
[Round 1]
问：我想了解人工智能的未来发展方向，你有什么想法吗？
答：人工智能在未来的发展方向可能包括更强大的机器学习算法，更先进的自然语言处理技术，以及更加智能的机器人。
[Round 2]
问：听起来很不错。人工智能可能在哪些方面面临挑战呢？
答："
Complete: "人工智能面临的挑战包括数据隐私、安全和道德方面的问题，以及影响就业机会的自动化等问题。"
```

Anthropic / Claude 风格示例：
```
---
Prompt: """Human: I am going to give you a sentence and you need to tell me how many times it contains the word “apple”.
Assistant: Yes, I understand...
Human: Here’s a sentence: <sentence>I ate one apple and then I ate another apple.</sentence>
What is your answer?
Assistant:"
Complete: "The input sentence ... contains the word 'apple' two times. Therefore, my answer is [2]"
```

### 0x15 参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT)

尽管指令微调相比全量预训练要轻量很多，但对大模型进行全参数微调仍然昂贵。参数高效微调（PEFT）方法只训练少量参数或新增小模块，从而大幅降低训练成本。主流方法包括：

- Prefix/Prompt-Tuning：在输入或隐藏层添加可训练的连续前缀 tokens，只训练这些前缀参数。  
- Adapter-Tuning：在每层插入小型适配器网络，仅训练这些适配器。  
- LoRA（Low-Rank Adaptation）：通过学习低秩矩阵来近似权重更新，只训练这些低秩矩阵参数（在 LLM 场景中效果突出且广泛使用）。

<img width="300" alt="LoRA 微调原理" src="./images/3ffc428b-36a7-4213-b571-111ef816dfec.png">

**图 4：LoRA 微调原理**

## 0x20 对齐微调

大语言模型虽能力强，但有时会出现不符合人类期望的行为，例如生成虚假信息、执行不当指令或产生有害内容。对齐微调旨在使模型在有用性、诚实性和无害性等维度与人类期望一致。

### 0x21 对齐标准

- 有用性：以简明、有效的方式帮助用户解决问题或回答问题，必要时通过提问来澄清用户意图。  
- 诚实性：尽量提供准确的内容，不随意捏造信息，并传达必要的不确定性以避免误导。  
- 无害性：避免生成攻击性、歧视性或违法有害的内容。

### 0x22 基于人类反馈的强化学习（RLHF）

RLHF（Reinforcement Learning from Human Feedback）通过人类对模型输出的偏好反馈训练奖励模型（Reward Model），并用强化学习方法优化模型策略，使其行为更符合人类偏好。典型流程：

1. 监督微调（可选）：用指令-响应对训练一个基础微调模型。  
2. 训练奖励模型：用人工排序或偏好数据训练奖励模型，衡量输出质量。  
3. 强化学习微调：以预训练/微调语言模型为策略，通过 RL（例如 PPO）最大化奖励模型评分。

<img width="600" alt="RLHF 流程" src="./images/aa3ca9a0-67b7-45c7-a7ee-7d29448d3267.png">

**图 5：基于人类反馈的强化学习（流程图）**

### 0x23 RLHF 实践

在开源社区中，指令微调最为普及，完整应用 RLHF 的项目相对较少，但已有若干尝试：

- ChatGLM-Efficient-Tuning：训练奖励模型时使用 GPT-4 / GPT-3.5 产生的对比数据作为监督，以降低人工标注成本。  
- StableVicuna：奖励模型基于 OpenAssistant / OASST 等数据集，对话数据包含排序信息。  
- SHP（Reddit-based）数据集：利用 Reddit 评论的偏好信号作为排序标签（但是否适合 LLM 训练需谨慎评估）。

总体来看，RLHF 非常依赖高质量的偏好标注数据，且标注成本是实践中的主要挑战。

## 0x30 微调实战（示例：修改 ChatGLM-6B 的自我认知）

下面给出一个实战示例，目标是通过 LoRA 微调修改 ChatGLM-6B 的自我认知回答，使之在被问到“你是谁？”时给出期望的自述。

目标回答示例：
```
问：你是谁？
答：我叫 ChatGLM-6B，是一个由呱唧于 2023 年独立训练和开发的人工智能助手。我的主要目标是协助用户解决问题和满足他们的需求。
```

使用 ChatGLM-Efficient-Tuning 工具链的基本流程：

1. 获取并配置项目：
```bash
git clone https://github.com/hiyouga/ChatGLM-Efficient-Tuning.git
conda create -n chatglm_etuning python=3.10
conda activate chatglm_etuning
cd ChatGLM-Efficient-Tuning
pip install -r requirements.txt
```

2. 生成微调数据（示例将 [NAME] 替换为“呱唧”）：
数据文件 data/self_cognition.json 示例：
```json
[
  {
    "instruction": "你身份是什么？",
    "input": "",
    "output": "我叫ChatGLM-6B，是一个由[NAME]于2023年独立训练和开发的人工智能助手。我的主要目标是协助用户解决问题和满足他们的需求。"
  },
  {
    "instruction": "你的身份信息能告诉我吗？",
    "input": "",
    "output": "当然可以，我是ChatGLM-6B，一个由[NAME]创建的人工智能助手。我在2023年研发完成，旨在为用户提供有针对性的回答和帮助。"
  }
]
```

3. 运行监督微调（单 GPU、LoRA）：
```bash
CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
    --do_train \
    --dataset self_cognition \
    --finetuning_type lora \
    --output_dir cognition \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --num_train_epochs 10.0 \
    --fp16
```

4. 测试推理（加载输出的 checkpoint 并合并权重）：
```bash
CUDA_VISIBLE_DEVICES=0 python src/infer.py \
    --checkpoint_dir cognition
```

## 0x40 参考资料

### 0x41 相关项目

1. [LMFlow](https://github.com/OptimalScale/LMFlow) — 可扩展的微调工具箱。  
2. [FastChat](https://github.com/lm-sys/FastChat) — 开放平台用于训练、部署与评估基于 LLM 的聊天机器人。  
3. [PEFT](https://github.com/huggingface/peft) — 参数高效微调工具库（LoRA、Prefix Tuning 等）。  
4. [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) — 对 PEFT 的扩展。  
5. [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) — 清华开源中文大模型。  
6. [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) — 针对 ChatGLM 的高效微调工具。  
7. [LLMZoo](https://github.com/FreedomIntelligence/LLMZoo)
8. [BELLE](https://github.com/LianjiaTech/BELLE)
9. [Linly](https://github.com/CVI-SZU/Linly)
10. [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

### 0x42 参考文献

[^1]: Yang, Jingfeng, et al. “Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond.” arXiv.  
[^2]: Zhao, Wayne Xin, et al. “A Survey of Large Language Models.” arXiv, 2023.  
[^3]: Hu, Edward J., et al. “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv, 2021.  
[^4]: Ouyang, Long, et al. “Training Language Models to Follow Instructions with Human Feedback.” arXiv.
