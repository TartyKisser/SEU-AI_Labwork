# base environments
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error
import re
import warnings

# personal packages
from optims import LinearWarmupCosineLRScheduler

warnings.filterwarnings("ignore")

# llm environments

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    T5Tokenizer,
    T5Model,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig, setup_chat_format


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        # the init will be impliment on cpu, which means self.device='cpu'. Then in training process, lightning will put the tensors and models into cuda automatically.
        self.cuda = 'cuda:' + self.hparams.cuda  # manually set cuda to avoid device bug
        self.load_llm()

        
    def forward(self, batch):
        input_pairs = [[prompt, answer] for prompt, answer in zip(batch['input'], batch['answer'])]
        input_encoding = self.tokenizer(input_pairs, return_tensors='pt', max_length=self.hparams.max_input_length, padding="max_length", truncation=True, return_token_type_ids=True)
        input_ids, attention_mask, token_type_ids = input_encoding.input_ids, input_encoding.attention_mask, input_encoding.token_type_ids
        input_embeds = self.model.get_input_embeddings()(input_ids.to(self.cuda))  # batch, max_length, dim

        # mask the input and pad tokens
        target_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        # target_ids = target_ids.masked_fill(token_type_ids == 0, -100)

        # device = input_embeds.device
        outputs = self.model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask.to(self.cuda),  # casual mask: next-token prediction
                    return_dict=True,
                    labels=target_ids.to(self.cuda),
                    use_cache=False,
                )
        lm_loss = outputs.loss

        return lm_loss
    
    
    def configure_loss(self, out, labels=None):
        loss = out  # you can build your losses here
        return loss
    
    
    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)  # use to check the anomaly in gradient backward
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss


    # validation functions
    def generate(self, batch, temperature=1, do_sample=False, num_beams=1, min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        max_gen_length = self.hparams.max_gen_length
        input_pairs = [prompt for prompt, answer in zip(batch['input'], batch['answer'])]
        input_encoding = self.tokenizer(input_pairs, return_tensors='pt', max_length=self.hparams.max_input_length, padding="max_length", truncation=True, return_token_type_ids=True)
        input_ids, attention_mask, token_type_ids = input_encoding.input_ids, input_encoding.attention_mask, input_encoding.token_type_ids

        outputs = self.model.generate(
            input_ids=input_ids.to(self.cuda),
            attention_mask=attention_mask.to(self.cuda),
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            return_legacy_cache=True,
            )
        
        sequence_ids = outputs['sequences']
        outputs = self.tokenizer.batch_decode(sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return outputs
    
    
    # -------------------- validation --------------------------
    def on_validation_epoch_start(self):
        self.val_content={
            "input":[],
            "generated_text":[],
            "label":[],
        }
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        texts = self.generate(batch)
        b = 0
        for prompt, answer in zip(batch['input'], batch['answer']):
            self.val_content['input'].append(prompt)
            self.val_content["label"].append(answer)
            self.val_content["generated_text"].append(texts[b])
            b += 1
          
            
    def on_validation_epoch_end(self):
        # calculate metric
        pattern = r'<(.*?)>'
        y_true = []
        y_pre = []
        for predict, label in zip(self.val_content['generated_text'], self.val_content['label']):
            predict = re.findall(pattern, predict)[-1]
            label = float(re.findall(pattern, label)[0])
            try:
                predict = float(predict)
            except:
                predict = 0
            y_pre.append(predict)
            y_true.append(label)
        mae = mean_absolute_error(y_true, y_pre)

        # save logs
        self.log('metric', mae, on_step=False, on_epoch=True, prog_bar=True)
    
    
    # -------------------- test ----------------------
    def on_test_epoch_start(self):
        self.test_content={
            "input":[],
            "generated_text":[],
            "label":[],
        }
    
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        texts = self.generate(batch)
        b = 0
        for prompt, answer in zip(batch['input'], batch['answer']):
            self.test_content['input'].append(prompt)
            self.test_content["label"].append(answer)
            self.test_content["generated_text"].append(texts[b])
            b += 1
            
            
    def on_test_epoch_end(self):
        # calculate metric
        pattern = r'<(.*?)>'
        y_true = []
        y_pre = []
        for predict, label in zip(self.test_content['generated_text'], self.test_content['label']):
            predict = re.findall(pattern, predict)[-1]
            label = float(re.findall(pattern, label)[0])
            try:
                predict = float(predict)
            except:
                predict = 0
            y_pre.append(predict)
            y_true.append(label)
        mae = mean_absolute_error(y_true, y_pre)

        # save logs
        self.log('metric', mae, on_step=False, on_epoch=True, prog_bar=True)
    

    def configure_optimizers(self):  # this function will run automatically
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                  max_step=max_step,
                                                  min_lr=self.hparams.lr_decay_min_lr,
                                                  init_lr=self.hparams.lr,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer
    
    
    # ------------------------ loading -------------------------
    def load_llm(self):
        # Llama Config
        model_name = 'Llama-3.2-1B-Instruct'  # model
        hf_token = "your_huggingface_token" # hf_token for Llama 3.1 or 3.2
        model_source = 'meta-llama/'
        model_id = model_source + model_name
        torch_dtype = torch.float16
        attn_implementation = "eager"  # eager, FlashAttention, ...
        cache_dir='/home/haohao/.huggingface'  # base model save_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # parameter quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            quantization_config=bnb_config,
            device_map=self.cuda,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        self.model = get_peft_model(model, peft_config)
        print('Loading LLAMA Done')