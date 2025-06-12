import argparse


# configurations
def parse_args():
    parser = argparse.ArgumentParser(description="Go ChatEV")
    
    # 
    parser.add_argument('--model', type=str, default='llama', help="the used language model")
    parser.add_argument('--test_only', default=False, action='store_true')
    
    # --------------- general --------------------
    parser.add_argument('--cuda', type=str, default='0', help="the used cuda")
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    
    # hyper
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    
    # llm
    parser.add_argument('--max_input_length', default=1024, type=int)
    parser.add_argument('--max_gen_length', default=32, type=int)
    
    # checkpoint
    parser.add_argument('--ckpt', default=False, action='store_true')
    parser.add_argument('--ckpt_name', default='last', type=str)
    
    # --------------- prediction ------------------------
    parser.add_argument('--data_name', default='occupancy', type=str)
    parser.add_argument('--zone', default=42, type=int)  
    parser.add_argument('--pre_len', default=6, type=int)
    parser.add_argument('--seq_len', default=12, type=int)
    
    # --------------- learning -------------------------
    parser.add_argument('--meta_learning', default=False, action='store_true')
    parser.add_argument('--outer_loop', default=10, type=int)
    parser.add_argument('--inner_loop', default=3, type=int)
    parser.add_argument('--few_shot', default=False, action='store_true')
    parser.add_argument('--few_shot_ratio', default=0.2, type=float, help="The first [0.05, 0.1, 0.15, 0.2] of the training data.")
    
    return parser.parse_args()