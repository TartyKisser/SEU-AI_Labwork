# ChatEV
This is a very simple implementation of **utilizing large language models (e.g., Llama-3.2-1B-Instruct) for time-series forecasting** in the scenarios of electric vehicle charging. If it is helpful to your research, please cite our papers:

>Haohao Qu, Han Li, Linlin You, Rui Zhu, Jinyue Yan, Paolo Santi, Carlo Ratti, Chau Yuen. (2024) ChatEV: Predicting electric vehicle charging demand as natural language processing. Transportation Research Part D: Transport and Environment. [Paper in TRD](https://doi.org/10.1016/j.trd.2024.104470)


### 1. Environments
For simple implementaton, we need five major packages, namely torch, pandas, numpy, transformers, and argparse. You can install these useful wheels by:

```shell
pip install -r requirements.txt
```

### 2. Meta-Llama hf_token
To get access to Meta-Llama models, we need to apply a "hf_token" key through https://huggingface.co/settings/tokens

Then replace input a correct "hf_token" in Line 99 of the "utils.py" file.
```shell
hf_token = "Your_HF_TOKEN"
```

ps: Or you can download a local model through https://www.llama.com/llama-downloads/

### 3. Simple Implementation
To conduct a simple implementation (inference only), we can run the "simple.py" file.
```shell
python simple.py
```

To conduct a simple finetuning implementation, we can run the "finetune.py" file.
```shell
python finetune.py
```

### 4. Full Implementation
All code for a complete implementation of ChatEV (including finetuning, validation, and testing) is included in the "code" folder. Besides the five packages for simple version, more environments are required for the full implementation: [argparse, lightning, scikit-learn].
```shell
pip install argparse, lightning, scikit-learn
```

**Please remember to change your path to the "code" folder.**
```shell
cd code
```

We can run the "main.py" file to fintune a Llama model for EV charging data prediction:
```shell
python main.py
```

### 5. Alternative Configurations
* If you wanna load a checkpoint for finetuning, you can do so:
```shell
python main.py --ckpt --ckpt_name='last'
```
* If you wanna load a checkpoint for testing, you can do so:
```shell
python main.py --ckpt --ckpt_name='last' --test_only
```
* Train the model in a few-shot scienario:
```shell
python main.py --few_shot --few_shot_ratio=0.2
```
* Train the model using a simple and effective meta-learning approach, First-Order Reptile:
```shell
python main.py --meta_learning
```

More configurations can be found in the "parse.py" file.

### 5. Questions
* If you uncounter a problem of slow downloading, you can set a mirror source at the command terminal:
```shell
export HF_ENDPOINT=https://hf-mirror.com
```
