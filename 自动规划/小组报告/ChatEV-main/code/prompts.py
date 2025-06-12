import numpy as np


# zone profile
def characterization(inf):
    id = inf.name
    la = inf['latitude']
    lon = inf['longitude']
    area = inf['area']
    cap = inf['charge_count']
    perimeter = inf['perimeter']
    des = f"""\nWe are now in Traffic Zone {id}. Its coordinates are ({la}, {lon}).
The zone covers an area of {area} square kilometres and has {cap} public charing piles. \n"""
    return des


# prompt template
def prompting(data, index, seq_len, pre_len, inf, data_name):
    np.set_printoptions(linewidth=1000)  # number of printed elements in a row.
    
    # history
    local_charge = np.around(np.array(data['local_charge'][index:index+seq_len]), decimals=4)
    neighbor_charge = np.around(np.array(data['neighbor_charge'][index:index+seq_len]), decimals=4)
    # current 
    local_prc = np.around(np.array(data['local_prc'][index+seq_len]), decimals=2)
    neighbor_prc = np.around(np.array(data['neighbor_prc'][index+seq_len]), decimals=2)
    temperature = np.array(data['temperature'][index+seq_len])
    humidity = np.array(data['humidity'][index+seq_len])
    # future
    f_prc = np.around(np.array(data['local_prc'][index+seq_len+pre_len]), decimals=2)
        
    # des = characterization(inf)
    
    template = f"""
    ### INPUT:
        You are an expert in electric vehicle charging management, who is good at charging demand prediction. 
        The weather is {temperature} degrees Celsius with a humidity of {humidity}.
        Given the following time series of historical charging data,
        Charging {data_name.title()} for the Previous {seq_len} hours = {local_charge};
        Charging Price (current|future) = {local_prc} | {f_prc}.
        Now, pay attention! Your task is to predict the charging {data_name} in the area for the next {pre_len} hour by analyzing the given information and leveraging your common sense.
        In your answer, you should provide the value of your prediction in angle brackets, such as <value>.
    ### RESPONSE:
    """
    return template


# output template
def output_template(data, data_name, future=6):
    data = str(data)
    prepend = dict()
    prepend[0] = f'The predicted value for the next {future} hours is <{data}>.'
    prepend[1] = f'The future charging {data_name} for the next {future} hours is <{data}>.'
    prepend[2] = f'I predict charging {data_name} for the next {future} hours to be approximately <{data}>.'
    idx = int(np.random.randint(0, len(prepend), 1))
    return prepend[idx]