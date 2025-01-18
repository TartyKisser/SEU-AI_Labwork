# About

A demo app for real-time task-by-task analysis of FSL algorithms. 

![GUI Screenshot](https://github.com/mattochal/demo_fsl_dev/blob/master/demo.gif?raw=true)

### Compatibility
This demo is compatible with the following GitHub repositories:
* [mattochal/imbalanced_fsl_public](https://github.com/mattochal/imbalanced_fsl_public)

### Requirements

1. Pretrained / meta-trained models weights. See the repositories above to train the models.
2. Dataset and repository copies on both the local and the remote machines.
3. [SSH tunneling](https://www.ssh.com/ssh/tunneling/), for example: 
    ```
    ssh -N -f -L localhost:8891:localhost:8891 user@serverwithmodels
    ```

### Structure

* ```server.py``` - resposible for loading and running the inference of models (on a remote server)
* ```client.py``` - resposible for displaying the GUI (on a local machine)
* ```server_args.json``` - configuration file for the server, containing directories and hyperparameters for models.


### Running the demo

On the remote server side (setup the SSH tunnel first!):
 1. Clone this repository into one of the compatible repositories.
 2. Fill out ```server_args.json``` with directories to pre-trained models (more details below). 
 3. Run ```server.py``` on the server side, e.g.:
    ```
    python server.py --args_file my_server_args.json --repo imbalanced_fsl --port 8891
    ```
    For full details see ```python server.py --help```
 4. Wait until you see the following message ```Waiting for a connection``` before running the client. 


On the client side (setup the remote server first!):
 1. Clone this repository into one of the compatible repositories (but this time on the client side!).
 2. Run ```client.py``` on the client side, for example:
    ```
    python client.py --port 8891 --data_path '../data/'
    ```
 3. Wait for the ```client.py``` to automatically get all necessary information from the server and load the GUI.


# Additional user guide

### Server args

Server args example:
```
{
    "port"       : 8991,
    "seed"       : 0,
    "dataset"    : "mini",
    "version"    : null,
    "data_path"  : "../data/",
    "exp_path"   : "../experiments/"
    "models":[
        {
            "name"    : "ProtoNet V1 on CPU",
            "gpu"           : "cpu",
            "continue_from" : "./protonet_v1/0/checkpoints/epoch-112"
        },
        {
            "model_name"    : "ProtoNet V2 on GPU",
            "gpu"           : "6",
            "continue_from" : "./protonet_v2/0/"
        }
        ...
    ]
}
```
Explanation:
 * ```port``` - to listen on
 * ```seed``` - for sampling and initializing the demo
 * ```dataset``` - name of the dataset
 * ```version``` - dataset version
 * ```data_path``` - directory containing the necessary data (see compatible repositories above)
 * ```exp_path``` - (optional) root experiment directory containing pretrained weights
 * ```models``` -  a list of models and specific directories to pretrained weights
    - ```name``` - (optional) memorable model name
    - ```gpu``` - device to load the model on ```cpu``` or a number. 
    - ```continue_from``` - folder or file path with model weights. If folder, uses the best model based on validation loss.

### GUI

![GUI Screenshot](https://github.com/mattochal/demo_fsl_dev/blob/master/screenshot.jpg?raw=true)

Screenshot showing 5-shot 5-way task. 

Actions:
1. Select support set. Click on images: red boundaries indicate current selection; blue boundaries indicate previous selections.
2. Shuffle support set candidates (images and classes).
3. Classify using the selected support set.
4. Performance Graphs. Top: bar chart of current task performace. Bottom: plot of all performances. Select the performance metric from the dropdown menu.
5. Confusion Matrix. Select the model from the dropdown menu. 
6. Query examples. Incorrect classifications indicated in brackets.
7. See other query examples.
8. Reset task. Reset the support set, query set, and model weights. 

### Client args

See ```python client.py --help``` for more details.

