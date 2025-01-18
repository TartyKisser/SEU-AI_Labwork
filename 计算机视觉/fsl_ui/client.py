import socket
import sys
import pickle
import time

from tkinter import *
from tkinter import ttk
from tkinter import messagebox, BooleanVar
from tkinter import font
from PIL import Image
import glob 
import os
from PIL import Image, ImageTk
import argparse
import pprint
import random
import numpy as np
import struct
import pdb
import pandas as pd 
import json


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib import style
style.use('ggplot')
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(os.path.dirname(currentdir),'src')
sys.path.insert(0,parentdir)

from utils.utils import get_args, get_data
 
BLANK_IMG = Image.new('RGB', (80, 80), (200,200,200)) 


def setup_connection(server_address=('localhost', 8990)):

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    print('connecting to {} port {}'.format(*server_address))

    sock.connect(server_address)
    return sock

def send_data(sock, data):
    print('sending', data)
    pickled = pickle.dumps(data)
    send_one_message(sock, pickled)

def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    time.sleep(0.5)
    sock.sendall(data)

def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    if not lengthbuf: return None
    length, = struct.unpack('!I', lengthbuf)
    data = recvall(sock, length)
    return data

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def wait_for_response(sock):
    print('waiting for a response')
    data = recv_one_message(sock)
    if data is None:
        return None
    return pickle.loads(data)


class Sampler():

    def __init__(self, dataset, seed, num_targets):
        self.dataset = dataset
        self.num_targets = num_targets
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.previous_supports = {}
        self.potential_supports = {}
        self.selected_supports = {}
        self.targets = {}

        for clss in self.dataset.class_dict:
            idxs = list(self.dataset.class_dict[clss])
            print(type(idxs))
            self.rng.shuffle(idxs)

            self.targets[clss] = idxs[:self.num_targets]
            self.potential_supports[clss] = idxs[self.num_targets:]
            self.previous_supports[clss] = []
            self.selected_supports[clss] = []

    def move_selected_to_previous(self):
        for clss in self.selected_supports:
            for idx in self.selected_supports[clss]:
                self.previous_supports[clss].append(idx)
            self.selected_supports[clss] = []

    def deselect_support(self, clss, idx):
        self.selected_supports[clss].remove(idx)
        self.potential_supports[clss].append(idx)

    def deselect_supports(self):
        for clss in self.selected_supports:
            for idx in self.selected_supports[clss]:
                self.potential_supports[clss].append(idx)
            self.selected_supports[clss] = []

    def select_support(self, clss, idx):
        self.selected_supports[clss].append(idx)
        self.potential_supports[clss].remove(idx)

    def sync_selected(self, selected):
        self.deselect_supports()
        for clss in selected:
            for idx in selected[clss]:
                self.select_support(clss, idx)

    def sample_potential_supports(self, n_by_clss=None):
        sample = {}
        for clss in self.potential_supports:
            potential = self.potential_supports[clss]
            selected = self.selected_supports[clss]

            if n_by_clss is None:
                n = len(potential) - len(selected)

            elif clss in n_by_clss:
                n = min(n_by_clss[clss], len(potential) - len(selected))

            elif clss not in n_by_clss:
                continue

            self.rng.shuffle(potential)
            count = 0
            sample[clss] = []

            for i in range(n):
                idx = potential[i]
                sample[clss].append(idx)

        return sample

    def get_previous_supports(self):
        return self.previous_supports

    def get_selected_supports(self):
        return self.selected_supports

    def get_targets(self, target_classes):
        return {clss:self.targets[clss] for clss in target_classes}

class MyButton(Button):
    def __init__(self, master, **kw):
        super().__init__(master, bg="red", activebackground="blue", **kw)
        # self.configure(bg="SlateBlue2", activebackground="SlateBlue1")
        # self.bind("<Enter>", self.on_enter)
        # self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        print("enter")
        # self.configure(bg="SlateBlue1")

    def on_leave(self, e):
        print("leave")
        # self.configure(bg="SlateBlue2", activebackground="SlateBlue2")


def _image_for_display(im):
    # print("type:", type(im))
    if type(im) == np.ndarray:
        h, w, c = np.shape(im)
        if c == 1:
            im = im.squeeze()
            im = Image.fromarray(im, 'L')
        else:
            im = Image.fromarray(im)
    im = im.resize((70, 70))
    im = ImageTk.PhotoImage(im)
    return im

class MyImageButton(Frame):

    def __init__(self, frame):
        super().__init__(frame, borderwidth=4)
        self.is_selected = False
        self.image = None
        self.enabled = True
        self.button = Button(self, padx=0, bd=0, pady=0, command=self.toggle)
        self.button.pack()
        self.stats = {}
        
    def toggle(self):
        if self.enabled:
            if self.is_selected:
                self.deselect()
            else:
                self.select()

    def set_image(self, image):
        self.image = _image_for_display(image)
        self.button.configure(image=self.image)

    def select(self):
        self.is_selected = True
        self.configure(bg="red")

    def deselect(self):
        self.is_selected = False
        self.configure(bg="white")

    def enable(self):
        self.enabled = True
        self.configure(bg="white")

    def disable(self):
        self.enabled = False
        self.configure(bg="light gray")

    def disable2(self):
        self.enabled = False
        self.configure(bg="blue")


class MyImageLabel(Frame):

    def __init__(self, frame):
        super().__init__(frame, borderwidth=4, padx=1, pady=1)
        self.is_selected = False
        self.image = None
        self.image_label = Label(self, padx=0, bd=0, pady=0, font=("Helvetica", 10))
        self.image_label.pack(fill=BOTH, expand=True)

        self.caption = None
        self.caption_label = Label(self,  padx=0, bd=0, pady=0, font=("Helvetica", 10))
        self.caption_label.pack(fill=BOTH, expand=True)

    def set_image(self, image):
        self.image = _image_for_display(image)
        self.image_label.configure(image=self.image)

    def set_caption(self, caption):
        self.caption = caption
        self.caption_label.configure(text=caption)

    def get_caption(self):
        return self.caption

    def set_style_colour(self, colour):
        self.configure(bg=colour)
        self.caption_label.configure(fg=colour)

class SupportSetButtonFrame(Frame):

    def __init__(self, master, num_rows, num_cols):
        super().__init__(master, borderwidth=4)
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        title_frame = Frame(self)
        title_frame.grid(column=0, row=0, sticky="wesn")
        title = Label(title_frame, text='Support set', borderwidth=5, font=("Helvetica", 10))
        title.pack(fill='both', expand=True)

        TITLE_FONT = font.Font(self, size=15)   
        title.configure(bg="gray", fg="white", font=TITLE_FONT)

        self.matrix_frame = Frame(self)
        self.matrix_frame.grid(column=0, row=1, sticky="swen")
        self.init_matrix()

        # self.control_frame = Frame(self)
        # resample_button_wrapper = Frame(self.control_frame)
        # resample_button_wrapper.grid(column=0, row=2, sticky="swen")
        # self.resample_button = Button(self.control_frame, text='Resample Supports')
        # self.resample_button.pack(fill=BOTH, expand=True)

    def clear_matrix(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

    def init_matrix(self):
        self.class_labels = []
        self.image_buttons = []

        for i_row in range(self.num_rows):
            label_wrapper = Frame(self.matrix_frame, borderwidth=2)
            label_wrapper.grid(column=0, row=i_row, sticky="swen")
            label = Label(label_wrapper, text="")
            label.configure(bg='#EEEEEE', fg='black')
            label.pack(fill='both', expand=True)
            self.class_labels.append(label)
            self.image_buttons.append([])

            for j_col in range(self.num_cols):
                button = MyImageButton(self.matrix_frame)
                button.set_image(BLANK_IMG)
                button.grid(column=j_col+1, row=i_row, sticky="swen")
                button.deselect()
                button.disable()
                self.image_buttons[i_row].append(button)

    def redraw_support_set(self, dataset, previous, selected, potential):
        classes = list()

        for cls in previous.keys():
            if len(previous[cls]) >= 1 and cls not in classes:
                classes.append(cls)
        
        # print('previous classes', classes)

        for cls in selected.keys():
            if len(selected[cls]) >= 1 and cls not in classes:
                classes.append(cls)
        
        # print('selected classes', classes)
        
        potential_classes = list(potential.keys())
        # print('before', potential_classes)
        random.shuffle(potential_classes)
        # print('after', potential_classes)
        for cls in potential_classes:
            if cls not in classes:
                classes.append(cls)
        
        print('classes', classes)

        self.image_button_mapping = []

        for i in range(min(len(classes), len(self.class_labels))):
            clss = classes[i]
            self.class_labels[i].configure(text=clss)
            self.image_button_mapping.append([])
            
            j_offset = 0

            if clss in previous:
                num_cols = min(len(previous[clss]), self.num_cols - j_offset)

                for j in range(num_cols):
                    button = self.image_buttons[i][j+j_offset]
                    idx = previous[clss][j]
                    image = dataset.image_data[idx]
                    button.set_image(image)
                    button.deselect()
                    button.disable2()

                    self.image_button_mapping[i].append((clss, idx))

                j_offset+=num_cols

            if clss in selected:
                num_cols = min(len(selected[clss]), self.num_cols - j_offset)

                for j in range(num_cols):
                    button = self.image_buttons[i][j+j_offset]
                    idx = selected[clss][j]
                    image = dataset.image_data[idx]
                    button.set_image(image)
                    button.enable()
                    button.select()

                    self.image_button_mapping[i].append((clss, idx))

                j_offset+=num_cols


            if clss in potential:
                num_cols = min(len(potential[clss]), self.num_cols - j_offset)

                for j in range(num_cols):
                    button = self.image_buttons[i][j+j_offset]
                    idx = potential[clss][j]
                    image = dataset.image_data[idx]
                    button.set_image(image)
                    button.enable()
                    button.deselect()
                    self.image_button_mapping[i].append((clss, idx))

                j_offset+=num_cols

            # fill up the remaining cols with blanks (i.e. when more columns than samples in the class)
            if self.num_cols - j_offset > 0:
                cols_remaining = self.num_cols - j_offset

                for j in range(cols_remaining):
                    button = self.image_buttons[i][j+j_offset]
                    button.set_image(BLANK_IMG)
                    button.deselect()
                    button.disable()
                    self.image_button_mapping[i].append(None)

        # fill up the remaining rows with blanks (i.e. when more rows than classes in dataset)
        for i in range(len(classes), self.num_rows):
            self.class_labels[i].configure(text="")
            self.image_button_mapping.append([])

            for j in range(self.num_cols):
                button = self.image_buttons[i][j]
                button.set_image(BLANK_IMG)
                button.deselect()
                button.disable()
                self.image_button_mapping[i].append(None)

    def add_row(self):
        self.num_rows += 1
        self.clear_matrix()
        self.init_matrix()

    def add_column(self):
        self.num_cols += 1
        self.clear_matrix()
        self.init_matrix()

    def subtract_row(self):
        self.num_rows -= 1
        self.clear_matrix()
        self.init_matrix()

    def subtract_column(self):
        self.num_cols -= 1
        self.clear_matrix()
        self.init_matrix()

    def get_selected(self):
        selected = {}
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                button = self.image_buttons[i][j]
                if button.enabled and button.is_selected:
                    tag = self.image_button_mapping[i][j]
                    if tag is not None:
                        clss, idx = tag
                        if clss not in selected:
                            selected[clss] = [idx]
                        else:
                            selected[clss].append(idx)
        return selected

    def deselect_all(self):
        selected = {}
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                button = self.image_buttons[i][j]
                if button.enabled and button.is_selected:
                    button.deselect()
        return selected


class GraphFrame(Frame):

    def __init__(self, master):
        super().__init__(master, borderwidth=4)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        title_frame = Frame(self)
        title_frame.grid(column=0, row=0, sticky="wesn")
        title = Label(title_frame, text='Graphs', borderwidth=5)
        title.pack(fill='both', expand=True, side = LEFT)
        TITLE_FONT = font.Font(self, size=15)   
        title.configure(bg="gray", fg="white", font=TITLE_FONT)

        graph_frame_wrapper = Frame(self)
        graph_frame_wrapper.grid(column=0, row=1, sticky="wesn")

        self.graph_frames = {}
        self.figs = {}
        self.graph_canvases = {}
        self.metrics = ['accuracy', 'precision', 'recall', 'f1']

        for i, m in enumerate(self.metrics):
            graph_frame = Frame(graph_frame_wrapper)
            graph_frame.grid(column=0, row=0, sticky="wesn")

            fig = Figure(figsize=(8.5,5), dpi=100)
            graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
            
            self.graph_frames[m] = graph_frame
            self.figs[m] = fig
            self.graph_canvases[m] = graph_canvas

            if i >= 1:
                graph_frame.grid_remove()
        
        dropdown_frame = OptionMenuFrame(title_frame, 'Metric', self.graph_frames, choice_order=self.metrics)
        dropdown_frame.pack(side = RIGHT)

    def plot(self, stats_by_subtask):
        for m in self.metrics:
            self.figs[m].clear()
            latest = stats_by_subtask[-1]
            models = []
            y = []
            for model in latest:
                models.append(model)
                print(latest[model].keys())
                y.append(latest[model][m])

            ax = self.figs[m].add_subplot(211)
            x = np.arange(len(models))
            ax.bar(x, y, 0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel(m)
            # ax.yticks(rotation=90)
            # ax.set_title('Accuracy for each algorithm')
            ax.yaxis.set_ticks(np.linspace(0, 1.0, 11))
            ax.set_ylim([0.15, 1.05])
            ax.tick_params(axis='y', which='major')

            for tick in ax.get_xticklabels():
                tick.set_rotation(10)

            ax = self.figs[m].add_subplot(212)
            # ax.set_title('Accuracy through the subtasks')
            x = list(range(len(stats_by_subtask)))
            for model in models:
                accs = []
                for stats in stats_by_subtask:
                    accs.append(stats[model][m])
                ax.plot(x, accs, marker='.')

            ax.yaxis.set_ticks(np.linspace(0, 1.0, 11))
            ax.set_ylim([0.15, 1.05])
            ax.tick_params(axis='y', which='major')
            ax.set_xticks(x)
            ax.set_ylabel(m)
            ax.set_xlabel('Subtask ID')
            ax.legend(models, loc='upper center', bbox_to_anchor=(0.5,1.27), ncol=4, frameon=True)
            plt.subplots_adjust(
                wspace=0.4, 
                hspace=0.155
            )
            self.figs[m].tight_layout()
            self.graph_canvases[m].draw()

    def clear_graph(self):
        for m, fig in self.figs.items():
            fig.clear()
        
        for m, graph in self.graph_canvases.items():
            graph.draw()


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


class OptionMenuFrame(Frame):

    def __init__(self, master, title, choice_frames, choice_order=None, grid_or_pack='grid'):
        super().__init__(master, borderwidth=4)

        if choice_order is None:
            choice_order = choice_frames.keys()

        wrapper_frame = Frame(self)
        wrapper_frame.grid(row = 0, column = 0, sticky="wesn")

        tkvar = StringVar(wrapper_frame)
        tkvar.set(choice_order[0])

        Label(wrapper_frame, text=title).grid(row = 0, column = 0)
        popupMenu = OptionMenu(wrapper_frame, tkvar, *choice_order)
        popupMenu.grid(row = 0, column = 1)

        # on change dropdown value
        def change_dropdown(*args):
            for algo_name in choice_frames:
                if grid_or_pack == 'grid':
                    choice_frames[algo_name].grid_remove() # remove from view
                else:
                    choice_frames[algo_name].pack_forget()
            algo_name = tkvar.get()
            if grid_or_pack == 'grid': 
                choice_frames[algo_name].grid()  # show in view
            else:
                choice_frames[algo_name].pack()

        # link function to change dropdown
        tkvar.trace('w', change_dropdown)


class ConfMatrixFrame(Frame):

    def __init__(self, master, models):
        super().__init__(master, borderwidth=4)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.models = models
        self.figs = {}
        self.graph_canvas = {}
        tab_frames = {}

        title_frame = Frame(self)
        title_frame.grid(column=0, row=0, sticky="wesn")
        title = Label(title_frame, text='Confusion Matrix (x=pred, y=true)', borderwidth=5)
        title.pack(fill='both', side = LEFT, expand=True)
        TITLE_FONT = font.Font(self, size=15)
        title.configure(bg="gray", fg="white", font=TITLE_FONT)

        for i, model in enumerate(models):
            model_tab = Frame(self)
            model_tab.grid(column=0, row=1, sticky="wesn")

            self.figs[model] = Figure(figsize=(8,3.5), dpi=100)
            self.graph_canvas[model] = FigureCanvasTkAgg(self.figs[model], master=model_tab)
            self.graph_canvas[model].draw()
            self.graph_canvas[model].get_tk_widget().pack(fill=BOTH, expand=True)

            tab_frames[model] = model_tab

            if i >= 1: 
                model_tab.grid_remove()
        
        dropdown_frame = OptionMenuFrame(title_frame, '', tab_frames, choice_order=models)
        dropdown_frame.pack(side = RIGHT)

    def clear_graph(self):
        for model in self.models:
            self.figs[model].clear()
            self.graph_canvas[model].draw()

    def plot(self, stats_by_subtask):

        latest = stats_by_subtask[-1]
        for model in latest:
            self.figs[model].clear()  
            ax = self.figs[model].add_axes([0.2,0.2,0.7,0.7])

            # set_size(3,3,ax)
            # ax.set_title('Confusion matrix ')

            conf_matrix = latest[model]['conf_matrix']
            conf_matrix_df = pd.DataFrame(conf_matrix)
            conf_matrix_df.sort_index(axis=0, inplace=True, ascending=True)
            conf_matrix_df.sort_index(axis=1, inplace=True, ascending=True)

            sns.heatmap(conf_matrix_df, annot=True, ax=ax, linewidths=0.5, annot_kws={"size": 12})
            ax.set_ylim(0, len(conf_matrix))
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor", fontsize=10)
            plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
            self.graph_canvas[model].draw()

class OutputSamplesTabFrame(Frame):

    def __init__(self, master, num_rows, num_cols):
        super().__init__(master)
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.imagelabels = []

        for i in range(num_rows):
            self.imagelabels.append([])
            for j in range(num_cols):
                label = MyImageLabel(self)
                label.set_image(BLANK_IMG)
                label.set_caption('n\\a')
                label.grid(column=j, row=i, sticky="swen")
                self.imagelabels[i].append(label)

    def clear_images(self):
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                image = BLANK_IMG
                label = "n\\a"
                colour = "white"
                self.imagelabels[r][c].set_image(image)
                self.imagelabels[r][c].set_caption(label)
                self.imagelabels[r][c].set_style_colour(colour)


    def draw_images(self, dataset, stats, seed, num_correct_rows):
        """
        num_correct_rows:
         -1: the correct and incorrect images proportional 
         n: top n rows for the correct, rest for incorrect 
         None: shuffled but the same for all algorithms
        """
        image_pairs = list(zip(stats['targets_idx'], stats["preds"], stats["true"]))
        rng = np.random.RandomState(seed)
        rng.shuffle(image_pairs)

        if num_correct_rows is None:
            count = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):

                    if count < len(image_pairs):
                        image = dataset.image_data[image_pairs[count][0]]
                        label = "{}\n({})".format(image_pairs[count][1], image_pairs[count][2])
                        colour = "green" if image_pairs[count][1] == image_pairs[count][2] else "red"
                    else:
                        image = BLANK_IMG
                        label = "n\\a"
                        colour = "white"

                    self.imagelabels[r][c].set_image(image)
                    self.imagelabels[r][c].set_caption(label)
                    self.imagelabels[r][c].set_style_colour(colour)

                    count += 1
        else:

            correct = []
            incorrect = []

            for i, pair in enumerate(image_pairs):
                if pair[1] == pair[2]:
                    correct.append(pair)
                else:
                    incorrect.append(pair)

            if num_correct_rows < 0:
                num_correct = int((len(correct) * self.num_rows * self.num_cols)/ (len(correct) + len(incorrect)))
            else:
                num_correct = self.num_cols * num_correct_rows

            num_correct = min(num_correct, len(correct))
            num_incorrect = len(incorrect)

            count = 0
            for r in range(self.num_rows):
                for c in range(self.num_cols):

                    if count < num_correct:
                        image = dataset.image_data[correct[count][0]]
                        label = "{}\n({})".format(correct[count][1], correct[count][2])
                        colour = "green"

                    elif count - num_correct < num_incorrect:
                        image = dataset.image_data[incorrect[count - num_correct][0]]
                        label = "{}\n({})".format(incorrect[count - num_correct][1], incorrect[count - num_correct][2])
                        colour = "red"

                    else:
                        image = BLANK_IMG
                        label = "n\\a"
                        colour = "white"

                    self.imagelabels[r][c].set_image(image)
                    self.imagelabels[r][c].set_caption(label)
                    self.imagelabels[r][c].set_style_colour(colour)
                    
                    count += 1


class OutputSamplesFrame(Frame):

    def __init__(self, master, models, num_rows, num_cols, num_correct_rows):
        super().__init__(master, borderwidth=4)
        self.models = models
        self.rng = np.random.RandomState(0)
        self.num_correct_rows = num_correct_rows
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        title_frame = Frame(self)
        title_frame.grid(column=0, row=0, sticky="wesn")
        title = Label(title_frame, text='Sample Outputs', borderwidth=5)
        title.pack(fill='both', expand=True, side=LEFT)

        TITLE_FONT = font.Font(self, size=15)   
        title.configure(bg="gray", fg="white", font=TITLE_FONT)
        
        self.tabs = {}
        for model in models:
            model_tab = OutputSamplesTabFrame(self, num_rows, num_cols)
            model_tab.grid(column=0, row=1, sticky="wesn")
            self.tabs[model] = model_tab

        dropdown_frame = OptionMenuFrame(title_frame, '', self.tabs, choice_order=models)
        dropdown_frame.pack(side=RIGHT)

    def draw_images(self, dataset, stats_by_subtask):
        seed = self.rng.randint(999999)
        latest = stats_by_subtask[-1]
        for model in self.tabs:
            self.tabs[model].draw_images(dataset, latest[model], seed, self.num_correct_rows)

    def clear_images(self):
        for model in self.tabs:
            self.tabs[model].clear_images()


class ControlFrame(Frame):

    def __init__(self, master):
        super().__init__(master, borderwidth=4)

        frame = Frame(self)
        frame.pack(fill=BOTH, expand=True)

        textfont = font.Font(self, size=15)  

        self.resample_supports_button = Button(frame, text='Resample Supports', command=self.resample_supports, font=textfont)
        self.resample_supports_button.pack(side=LEFT, expand=True)

        # self.deselect_button = Button(frame, text='Deselect', command=self.deselect_all)
        # self.deselect_button.pack(side=LEFT)

        # self.select_button = Button(frame, text='Select All', command=self.select_all)
        # self.select_button.pack(side=LEFT)

        # self.select_random_button = Button(frame, text='Select 5 Random Rows', command=self.select_5_random_rows)
        # self.select_random_button.pack(side=LEFT)

        self.classify_button = Button(frame, fg='green', text='Classify', command=self.classify,  font=textfont, width=50)
        self.classify_button.pack(side=LEFT, expand=True)

        self.resample_output_button = Button(frame, text='Resample Output', command=self.resample_outputs,  font=textfont)
        self.resample_output_button.pack(side=LEFT, expand=True)

        self.reset_button = Button(frame, fg='red', text='Reset Task', command=self.reset_all,  font=textfont)
        self.reset_button.pack(side=RIGHT)

    def resample_supports(self):
        print("ControlFrame.resample_supports")
        pass

    def deselect_all(self):
        pass

    def select_all(self):
        pass

    def classify(self):
        pass

    def resample_outputs(self):
        pass

    def reset_all(self):
        pass

    def disable_buttons(self):
        self.resample_supports_button['state'] = 'disable'
        self.classify_button['state'] = 'disable'
        # self.select_button['state'] = 'disable'
        # self.deselect_button['state'] = 'disable'
        # self.select_random_button['state'] = 'disable'
        self.resample_output_button['state'] = 'disable'
        self.reset_button['state'] = 'disable'

    def enable_buttons(self):
        self.resample_supports_button['state'] = 'normal'
        self.classify_button['state'] = 'normal'
        # self.select_button['state'] = 'normal'
        # self.deselect_button['state'] = 'normal'
        # self.select_random_button['state'] = 'normal'
        self.resample_output_button['state'] = 'normal'
        self.reset_button['state'] = 'normal'


class MainWindow(Frame):

    def __init__(self, master, args):
        super().__init__(master)
        self.args = args
        self.stats = []
        self.master = master
        # self.master.pack(expand=True, fill=BOTH)
        self.output_samples_on = (args.num_output_rows > 0) and (args.num_output_columns > 0)

    def initUI(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        
        self.support_frame = SupportSetButtonFrame(self.master, self.args.max_num_classes, self.args.max_labelled_num_images_per_class)
        self.support_frame.grid(column=0, row=0, sticky="wesn")

        center_frame = Frame(self.master)
        center_frame.grid(column=1, row=0, sticky="wesn")
        # center_frame.rowconfigure(0, weight=1)
        # center_frame.rowconfigure(1, weight=1)
        # center_frame.columnconfigure(0, weight=1)
        center_frame.configure(bg='black')

        self.graph_frame = GraphFrame(center_frame)
        self.graph_frame.grid(column=0, row=0, sticky="wesn")
        self.conf_matrix_frame = ConfMatrixFrame(center_frame, self.algorithm_names)
        self.conf_matrix_frame.grid(column=0, row=1, sticky="wesn")

        if self.output_samples_on:
            self.output_samples_frame = OutputSamplesFrame(self.master, self.algorithm_names, self.args.num_output_rows, self.args.num_output_columns, self.args.num_correct_output_rows)
            self.output_samples_frame.grid(column=2, row=0, sticky="wesn")

        self.control_frame = ControlFrame(self.master)
        self.control_frame.grid(column=0, row=1, sticky="wesn", columnspan=3)

        self.control_frame.resample_supports_button.configure(command=self.resample_supports)
        # self.control_frame.deselect_all = self.deselect_all
        # self.control_frame.select_all = self.select_all
        self.control_frame.classify_button.configure(command=self.classify)
        self.control_frame.resample_output_button.configure(command=self.resample_outputs)
        self.control_frame.reset_button.configure(command=self.reset_all)

    def init_with_server(self):
        print('setting up connection with server')
        address = ('localhost', self.args.port)
        self.sock = setup_connection(address)

        print('syncing with server')
        data = {'action':'setup'}
        send_data(self.sock, data)
        message = self.wait_for_response(loud=False)

        assert message['action'] == 'setup'

        self.algorithm_names = message['algorithms']
        self.num_algorithms = len(self.algorithm_names)
        
        message['dataset_args']['train']['data_path'] = self.args.data_path
        message['dataset_args']['val']['data_path'] = self.args.data_path
        message['dataset_args']['test']['data_path'] = self.args.data_path

        args, excluded_args, parser = get_args(stdin_list=['--dataset', message['dataset'], 
                                               '--dataset_args', json.dumps(message['dataset_args'])])

        print(args)
        datasets = get_data(args)
        self.dataset = datasets['test']

        assert message['dataset_sig'] == self.dataset.get_signature(), 'The datasets files and dataset_args must match!'

        self.sampler = Sampler(self.dataset, np.random.randint(99999), self.args.num_targets)

    def resample_supports(self):
        # print("MainFrame.resample_supports")
        selected = self.support_frame.get_selected()
        self.sampler.sync_selected(selected)

        previous = self.sampler.get_previous_supports()
        selected = self.sampler.get_selected_supports()
        potential = self.sampler.sample_potential_supports()

        self.support_frame.redraw_support_set(self.dataset, previous, selected, potential)
        pass

    def classify(self):
        selected = self.support_frame.get_selected()
        self.sampler.sync_selected(selected)

        selected = self.sampler.get_selected_supports()
        previous = self.sampler.get_previous_supports()
        classes  = [clss for clss in selected if len(selected[clss]) > 0]
        classes += [clss for clss in previous if len(previous[clss]) > 0]
        targets = self.sampler.get_targets(set(classes))

        selected_idx  = [idx for clss in previous for idx in previous[clss]]
        selected_idx += [idx for clss in selected for idx in selected[clss]]

        if len(selected_idx) == 0:
            messagebox.showerror("Error", "No supports selected")
            return

        target_pairs = [(idx,clss) for clss in targets for idx in targets[clss]]
        targets_idx = [ pair[0] for pair in target_pairs ]
        target_labels = [ pair[1] for pair in target_pairs ]

        print("support", selected_idx)
        print("support labels:", set(classes))
        print("target", targets_idx)
        print("target labels:", set(target_labels))

        message = {
            'action': 'classify',
            'supports': selected_idx,
             # [3163, 3166, 1507, 1512, 1502, 
                        # 3283, 3286, 1627, 1632, 1622, 
                        # 3193, 3196, 1537, 1542, 1532, 
                        # 3253, 3256, 1597, 1602, 1592, 
                        # 1746, 2990, 112, 2128, 2199], # selected_idx,
            'targets': targets_idx
            # [3155, 1517, 3152, 1526, 1508, 1525, 3168, 3169, 3173, 3164, 1528, 1520, 1527, 1503, 3159, 
                        # 3275, 1637, 3272, 1646, 1628, 1645, 3288, 3289, 3293, 3284, 1648, 1640, 1647, 1623, 3279, 
                        # 3185, 1547, 3182, 1556, 1538, 1555, 3198, 3199, 3203, 3194, 1558, 1550, 1557, 1533, 3189, 
                        # 3245, 1607, 3242, 1616, 1598, 1615, 3258, 3259, 3263, 3254, 1618, 1610, 1617, 1593, 3249, 
                        # 2326,  333, 2087, 2250,  229,  201, 1085, 3072, 1352,  892, 1831, 1436,  391, 2411,  374]  # targets_idx
            }

        send_data(self.sock, message)
        message = self.wait_for_response(loud=True)

        if message is None:
            messagebox.showerror("Error", "Server failed")
            return

        assert message['action'] == 'output'

        stats = {}
        for model_name in self.algorithm_names:
            model_stats = {}
            model_stats['preds'] = message['models'][model_name]
            model_stats['true'] = target_labels
            model_stats['targets_idx'] = targets_idx

            # acc = 0
            conf_matrix = { label1:{ label2: 0 for label2 in target_labels} for label1 in target_labels}

            for pred, true in zip(model_stats['preds'], model_stats['true']):
                # acc += 1 if pred == true else 0
                if pred not in conf_matrix:
                    conf_matrix[pred] = { label: 0 for label in target_labels }
                conf_matrix[pred][true] += 1
            # acc = acc * 1. / len(targets_idx)
            
            output = precision_recall_fscore_support(model_stats['true'], model_stats['preds'], beta=1.0)
            precision=output[0]
            recall=output[1]
            f1=output[2]
            acc=accuracy_score(model_stats['true'], model_stats['preds'])

            model_stats['accuracy']=acc
            model_stats['conf_matrix']=conf_matrix
            model_stats['precision']=precision.mean()
            model_stats['recall']=recall.mean()
            model_stats['f1']=f1.mean()

            stats[model_name] = model_stats

        self.stats.append(stats)

        self.draw_stats()
        self.resample_outputs()
        self.next_episode()

    def next_episode(self):
        self.sampler.move_selected_to_previous()
        self.support_frame.deselect_all()
        self.resample_supports()

    def draw_stats(self):
        self.graph_frame.plot(self.stats)
        self.conf_matrix_frame.plot(self.stats)

    def reset_all(self):
        self.graph_frame.clear_graph()
        self.conf_matrix_frame.clear_graph()
        if self.output_samples_on:
            self.output_samples_frame.clear_images()
        self.stats = []
        self.support_frame.deselect_all()
        self.sampler = Sampler(self.dataset, np.random.randint(99999), self.args.num_targets) 
        self.resample_supports()
        message = {'action':'reset_task'}
        send_data(self.sock, message)

    def resample_outputs(self):
        if len(self.stats) == 0:
            messagebox.showerror("Error", "Press 'Classify'")
            return
        if self.output_samples_on:
            self.output_samples_frame.draw_images(self.dataset, self.stats)

    def wait_for_response(self, loud=False):
        print('waiting for a response')
        if loud:
            self.control_frame.disable_buttons()
            messagebox.showerror("Info", "Wait for results from the server")
        data = wait_for_response(self.sock)
        if loud:        
            self.control_frame.enable_buttons()
        return data

def main(args):
    if args.task == 'fsl':
        root = Tk()
        root.geometry("2250x1000")
        app = MainWindow(root, args)
        app.init_with_server()
        app.initUI()
        app.resample_supports()
        root.mainloop()
    else:
        raise NotImplementedError()

def get_demo_parser(parser=argparse.ArgumentParser()):
    parser.add_argument('--max_num_classes', default=11)
    parser.add_argument('--max_labelled_num_images_per_class', type=int, default=6,
                        help='max number of images available for labeling')
    parser.add_argument('--num_output_columns', type=int, default=3,
                        help='Number of columns to displays for output images')
    parser.add_argument('--num_output_rows', type=int, default=8,
                        help='Number of incorrectly labellled images to display')
    parser.add_argument('--num_correct_output_rows', default=None,
                        help='Number of correctly labellled images to display')
    parser.add_argument('--checkpoint_name', default=None)
    parser.add_argument('--experiment_folder', default=None)
    parser.add_argument('--num_targets', default=15)
    parser.add_argument('--port', type=int, default=8891)
    parser.add_argument('--task', default='fsl', type=str)
    parser.add_argument('--data_path',  type=str, default="/Users/mateuszochal/Documents/University/PhD/datasets/",
                        help="Path to folder with all datasets.")
    return parser

def get_demo_args():
    parser = get_demo_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    pprint.pprint(args_dict, indent=4)
    args = Bunch(args_dict)
    return args

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
    def update(self, adict):
        self.__dict__.update(adict)

if __name__ == '__main__':
    args = get_demo_args()
    main(args)
    