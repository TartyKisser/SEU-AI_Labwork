import pandas as pd
import numpy as np
import torch, os, argparse


# packages for LLMs
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

# input datas
def read_data():
    occ = pd.read_csv('./data/occupancy.csv', index_col=0, header=0)  # occupancy ratio
    dur = pd.read_csv('./data/duration.csv', index_col=0, header=0)
    vol = pd.read_csv('./data/volume.csv', index_col=0, header=0)
    inf = pd.read_csv('./data/inf.csv', index_col=None, header=0)
    e_prc = pd.read_csv('./data/e_price.csv', index_col=0, header=0)  # electricity price
    s_prc = pd.read_csv('./data/s_price.csv', index_col=0, header=0)  # service price
    adj = pd.read_csv('./data/adj_filter.csv', index_col=0, header=0)  # adjacency matrix
    dis = pd.read_csv('./data/zone_dist.csv', index_col=0, header=0)  # distance between zones
    weather = pd.read_csv('./data/weather_central.csv', index_col=0, header=0)  # weather data

    col = occ.columns  # headers
    # occ = np.array(occ, dtype=float)
    # dur = np.array(dur, dtype=float)
    # vol = np.array(vol, dtype=float)
    prc = s_prc + e_prc
    adj = np.array(adj, dtype=float)
    dis = np.array(dis, dtype=float)
    # weather = np.array(weather)
    return occ, dur, vol, prc, adj, col, dis, weather, inf


# zone profile
def characterization(inf, n):
	id = inf['grid'][n]
	la = inf['la'][n]
	lon = inf['lon'][n]
	area = inf['area'][n]
	cap = inf['count'][n]
	des = f"""Traffic Zone {id}
	Its coordinates are ({la}, {lon}).
	The zone covers an area of {area} square kilometres and has {cap} public charing piles"""
	return des


# prompt template
def prompting(zone, timestamp, inf, data, prc, weather, length=12, future=6):
    np.set_printoptions(linewidth=1000)  # number of printed elements in a row.
    # des = characterization(inf, zone)
    occ = str(np.around(np.array(data.iloc[timestamp-length:timestamp, zone]), decimals=4))
    c_prc = str(np.around(prc.iloc[timestamp, zone], decimals=4))
    f_prc = str(np.around(prc.iloc[timestamp+future, zone], decimals=4))
    temperature = str(weather['T'].iloc[timestamp])
    humidity = str(weather['U'].iloc[timestamp])
    template = f"""
    ### INPUT:
        You are an expert in electric vehicle charging management, who is good at charging demand prediction. 
        We are now in Zone {str(zone)}.
        The weather is {temperature} degrees Celsius with a humidity of {humidity}.
        Given the following time series of historical charging data,
        Charging Occupancy for the Previous {length} hours = {occ};
        Charging Price (current|future) = {c_prc} | {f_prc}.
        Now, pay attention! Your task is to predict the charging occupancy in the area for the next {future} hour by analyzing the given information and leveraging your common sense.
        In your answer, you should provide the value of your prediction only.
    ### RESPONSE:
    """
    return template


# output template
def output_template(data, future=6):
    data = str(data)
    prepend = dict()
    prepend[0] = f'The predicted value for the next {future} hours is {data}.'
    prepend[1] = f'The future charging occupancy for the next {future} hours is {data}.'
    prepend[2] = f'I predict Charging Occupancy for the next {future} hours to be approximately {data}.'
    idx = int(np.random.randint(0, len(prepend), 1))
    return prepend[idx]


def load_llm(model_id='meta-llama/Llama-3.2-1B-Instruct', peft=False):
    # ----------------------------- Load model -----------------------------------
    # config
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # mirror source for hugging face
    hf_token = "input your own token" # hf_token for Llama 3.1 or 3.2
    torch_dtype = torch.float16
    attn_implementation = "eager"
    cache_dir= '/mnt/e/autoplan/ChatEV-main/model'

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        
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
        device_map="auto",
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
        local_files_only=True
    )

    if peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )

        model = get_peft_model(model, peft_config)
        

    parser = argparse.ArgumentParser(description="Generation Config")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--do_sample', default=False, action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--min_gen_length', type=int, default=1)
    parser.add_argument('--max_gen_length', type=int, default=128)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    
    return model, tokenizer, parser.parse_args()


