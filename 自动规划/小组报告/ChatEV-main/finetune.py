# Foundation packages
import torch
import pandas as pd
import numpy as np

# packages for data processing
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")


# input data
occ, dur, vol, prc, adj, col, dis, weather, inf = utils.read_data()
zone = 42
timestamp = 1024
input_prompt = utils.prompting(zone, timestamp, inf, occ, prc, weather)
target = utils.output_template(np.round(occ.iloc[timestamp+6, zone], decimals=4))
print(input_prompt)

model, tokenizer, config = utils.load_llm(peft=True)  # load model and tokenizer

# create LLM input
input_encoding = tokenizer([[input_prompt, target]], return_tensors='pt', max_length=1024, padding="max_length", truncation=True, return_token_type_ids=True)
input_ids, attention_mask, token_type_ids = input_encoding.input_ids, input_encoding.attention_mask, input_encoding.token_type_ids
input_embeds = model.get_input_embeddings()(input_ids.to(device))  # ids to embeds

# mask the input and pad tokens so that they are excluded from the loss/gradient computation.
target_ids = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
target_ids = target_ids.masked_fill(token_type_ids == 0, -100)  # 0:input_prompt, 1:target

outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(device),
            return_dict=True,
            labels=target_ids.to(device),
            use_cache=False,
)
lm_loss = outputs.loss
print(lm_loss)
