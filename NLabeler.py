import pygame as pg
from pygame.locals import *
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfile
from PIL import Image, ImageTk
import numpy as np
import scipy.misc
import os
import cv2 as cv
import warnings

class NLabeler(tk.Tk):
    def __init__(self, parent=None):
        #self.root = tk.Tk()
        tk.Tk.__init__(self, parent)
        self.title("NLabeler")
        self.parent = parent
        
        self.bind("<Left>", lambda event: self.next_frame(event, frames=-1))
        self.bind("<Shift-Left>", lambda event: self.next_frame(event, frames=-10))
        self.bind("<Right>", lambda event: self.next_frame(event, frames=1))
        self.bind("<Shift-Right>", lambda event: self.next_frame(event, frames=10))
        self.bind("<Down>", lambda event: self.next_frame(event, frames=-100))
        self.bind("<Shift-Down>", lambda event: self.next_frame(event, frames=-1000))
        self.bind("<Up>", lambda event: self.next_frame(event, frames=100))
        self.bind("<Shift-Up>", lambda event: self.next_frame(event, frames=1000))
        self.bind("<Delete>", self.clear_segment)
        
        self.image_width = 640
        self.image_height = 480
        self.fps = 0

        self.data_width = 64
        self.data_height = 64

        modes = ('record', 'label')
        self.mode = ''
        
        self.is_recording = False
        self.label_mode = ''

        self.capture = None

        self.current_frame = np.zeros((640,480, 3))
        self.frames = np.zeros((1,640,480))
        self.current_frame_n = 0
        self.total_frames = 1

        self.segment_windows = [SegmentWindow(self.image_width, self.image_height, self.data_width, self.data_height)]

        self.button1_isdown = False

        self.class_labels = ('A','B','C','D','E','F','G',
                             'H','I','J','K','L','M','N',
                             'O','P','Q','R','S','T','U',
                             'V','W','X','Y','Z','0')

        ### Create toolbar ###
        toolbar = tk.Menu(self)
        
        menu_file = tk.Menu(toolbar)
        menu_file.add_command(label="Open...", command=self.openfile_callback)
        
        menu_options = tk.Menu(toolbar)
        options_mode = tk.Menu(menu_options)
        options_mode.add_command(label="Record mode", command=self.set_record_mode)
        options_mode.add_command(label="Label mode", command=self.set_label_mode)
        menu_options.add_cascade(label="Mode", menu=options_mode)
        
        toolbar.add_cascade(label="File", menu=menu_file)
        toolbar.add_cascade(label="Options", menu=menu_options)
        self.config(menu=toolbar)

        ### Create parent frames ###
        parent_frame_image = tk.Frame(self)
        parent_frame_image.pack(side=tk.LEFT)

        parent_frame_inspector = tk.Frame(self)
        parent_frame_inspector.pack()

        ### Create sub frames ###
        self.frame_image = tk.Frame(parent_frame_image, width=self.image_width, height=self.image_height)
        self.frame_image.pack()

        self.frame_image.bind("<Button-1>", self.mouse_button1_press)
        self.frame_image.bind("<ButtonRelease-1>", self.mouse_button1_release)
        self.frame_image.bind("<Motion>", self.mouse_motion)

        self.inspector_frames = {}
        for mode in modes:
            self.inspector_frames[mode] = tk.Frame(parent_frame_inspector, width=160, height=self.image_height)
            self.inspector_frames[mode].pack_propagate(0)
            self.inspector_frames[mode].grid(row=0, column=0)

        ### Create widgets ###
        parent = self.inspector_frames['record']
        
        record_title = tk.Label(parent, text="Recording Mode")
        record_title.pack()

        self.btn_start_recording = tk.Button(parent, text="Start recording", command=self.start_recording)
        self.btn_start_recording.pack()

        self.btn_stop_recording = tk.Button(parent, text="Stop recording", command=self.stop_recording)
        self.btn_stop_recording.pack()

        parent = self.inspector_frames['label']

        label_title = tk.Label(parent, text="Labeling Mode")
        label_title.pack()

        self.var_frame_current = tk.StringVar()

        label_frame_current = tk.Label(parent, textvariable=self.var_frame_current)
        label_frame_current.pack()

        frame_image_nav = tk.Frame(parent)
        frame_image_nav.pack()
        btn_prev_frame = tk.Button(frame_image_nav, text="Prev", command=lambda: self.next_frame(frames=-1))
        btn_prev_frame.pack(side=tk.LEFT)
        btn_next_frame = tk.Button(frame_image_nav, text="Next", command=lambda: self.next_frame(frames=1))
        btn_next_frame.pack(side=tk.LEFT)
        
        btn_clear_segment = tk.Button(parent, text="Clear segment", command=self.clear_segment)
        btn_clear_segment.pack()

        self.canvas_segment = tk.Canvas(parent, width=128, height=128)
        self.canvas_segment.pack()
        
        array = scipy.misc.imresize(np.random.rand(32,32), (128,128), interp='nearest')
        self.segment_image = ImageTk.PhotoImage(Image.fromarray(array.astype('uint8')))
        self.window_image = self.canvas_segment.create_image((64, 64), image=self.segment_image)

        self.var_class_label = tk.StringVar()
        self.var_class_label.set("Default")
        label_class_label = tk.Label(parent, textvariable=self.var_class_label)
        label_class_label.pack()

        ### Frame for the class selection widget
        self.frame_class_selector = tk.Frame(parent)
        self.frame_class_selector.pack()

        self.var_class_label_value = tk.IntVar()
        self.init_class_selector(self.frame_class_selector, self.class_labels)

        # Save button
        btn_save = tk.Button(parent, text="Save", command=self.savefile_callback)
        btn_save.pack()

        os.environ['SDL_WINDOWID'] = str(self.frame_image.winfo_id())
        self.update()

        pg.init()

        self.screen = pg.display.set_mode((self.image_width, self.image_height))
        self.clock = pg.time.Clock()

        self.set_label_mode()

        self.mainloop()

    def mainloop(self):
        while True:
            self.screen.fill(pg.Color(0,0,0))

            if self.mode == 'record':
                _, frame = self.capture.read()
                frame = frame.transpose((1,0,2))
                frame = np.flip(frame, axis=0)
                frame = np.flip(frame, axis=2)
                self.current_frame = frame

                if self.is_recording:
                    self.frames.append(frame)

            pg.surfarray.blit_array(self.screen, self.current_frame)

            if self.mode == 'label':
                if self.segment_windows[self.current_frame_n].is_active:
                    pg.draw.rect(self.screen, pg.Color(0,0,255),
                                 self.segment_windows[self.current_frame_n].selection_window, 2)

                if self.button1_isdown:
                    pg.draw.rect(self.screen, pg.Color(0,127,0),
                                 self.segment_windows[self.current_frame_n].get_resized_window(self.mouse_pos), 1)

                    pg.draw.rect(self.screen, pg.Color(0,255,0),
                                 self.segment_windows[self.current_frame_n].get_mouse_window(self.mouse_pos), 2)
            
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)

    def open_npy(self, path):
        self.label_mode = 'numpy'

        self.frames = np.load(path).astype('uint8').transpose(0,2,1,3)
        self.total_frames = len(self.frames)
        self.current_frame_n = 0
        self.set_image_size(self.frames.shape[1], self.frames.shape[2])

    def open_avi(self, path):
        print(path)
        self.label_mode = 'avi'

        self.capture = cv.VideoCapture(path)
        self.total_frames = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))
        print("Opened a .avi file of %i frames"%self.total_frames)
        self.current_frame_n = 0

        ret, frame = self.capture.read()
        self.set_image_size(frame.shape[1], frame.shape[0])

    def set_record_mode(self):
        if self.mode != 'record':
            print("Set mode to recording")
            self.set_mode('record')

    def set_label_mode(self):
        if self.mode != 'label':
            print("Set mode to labeling")
            self.set_mode('label')

    def set_mode(self, mode):
        #prev_mode = self.mode
        self.mode = mode

        self.inspector_frames[mode].tkraise()

        if self.capture:
            self.capture.release()

        if mode == 'record':
            self.set_image_size(640, 480)
            self.capture = cv.VideoCapture(1)

    def update_frame(self):
        if self.mode == 'label':
            # Update frame counter
            self.var_frame_current.set("%i/%i"%(self.current_frame_n+1, self.total_frames))

            # Update class label
            label = self.segment_windows[self.current_frame_n].label
            if label < 0:
                self.var_class_label.set("Default")
            else:
                self.var_class_label.set(self.class_labels[self.segment_windows[self.current_frame_n].label])
            
            if self.label_mode == 'numpy':
                self.current_frame = self.frames[self.current_frame_n]
            elif self.label_mode == 'avi':
                self.capture.set(cv.CAP_PROP_POS_FRAMES, self.current_frame_n)
                ret, frame = self.capture.read()

                if ret:
                    frame = frame.transpose((1,0,2))
                    frame = np.flip(frame, axis=2)
                    self.current_frame = frame

            self.set_segment(not self.segment_windows[self.current_frame_n].is_active)

    def set_image_size(self, width=640, height=480):
        self.image_width = width
        self.image_height = height
        self.frame_image.config(width=width,
                                height=height)
        self.screen = pg.display.set_mode((width, height))

    def get_mouse_pos(self, event):
        self.mouse_pos = (event.x, event.y)

    def set_segment(self, clear=False):
        if clear:
            segment_array = np.zeros((128,128,3))
        else:
            segment_array = self.segment_windows[self.current_frame_n].get_segment(self.current_frame)
            segment_array = scipy.misc.imresize(segment_array, (128,128), interp='nearest')
        
        self.segment_image = ImageTk.PhotoImage(Image.fromarray(segment_array.astype('uint8')))
        self.canvas_segment.itemconfig(self.window_image, image=self.segment_image)

    ## Callback functions ##
    def openfile_callback(self):
        print("Open file")
        path = askopenfilename(filetypes=[("Any filetype", "*.*"),
                                          ("Numpy files", "*.npy"),
                                          ("Avi", "*.avi")])

        self.set_mode('label')
        
        if ".npy" in path:
            self.open_npy(path)
        elif ".avi" in path:
            self.open_avi(path)
        elif path == ():
            return
        else:
            warnings.warn("Unknown filetype")
            return

        self.segment_windows = [SegmentWindow(self.image_width,
                                    self.image_height,
                                    self.data_width, self.data_height) for i in range(self.total_frames)]

        self.update_frame()

    def savefile_callback(self):
        data = []
        labels = []
        
        for i in range(self.total_frames):
            if self.segment_windows[i].is_active:
                if self.label_mode == 'numpy':
                    data.append(self.segment_windows[i].get_segment(self.frames[i]))
                elif self.label_mode == 'avi':
                    self.capture.set(cv.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.capture.read()

                    if ret:
                        frame = frame.transpose((1,0,2))
                        frame = np.flip(frame, axis=2)
                        data.append(self.segment_windows[i].get_segment(frame))

                labels.append(self.segment_windows[i].label)

        path = asksaveasfile(mode='w').name
        data_file = path + "_data"
        label_file = path + "_labels"

        print(label_file)
        
        np.save(data_file, data)
        np.save(label_file, labels)
        print("Saved " + str(len(data)) + " frames to: " + data_file + ".npy and " + str(len(labels)) + " to: " + label_file)

        print(np.load(str(label_file) + ".npy"))
        
    def start_recording(self, event=None):
        print("Start recording")
        self.frames = []
        
        self.fps = 5
        self.is_recording = True

    def stop_recording(self, event=None):
        self.frames = np.asarray(self.frames, dtype='uint8')
        
        self.fps = 60
        self.is_recording = False

        print("Recorded %i frames"%len(self.frames))
        
    def mouse_button1_press(self, event):
        self.button1_isdown = True
        self.get_mouse_pos(event)
        self.segment_windows[self.current_frame_n].press_mouse(self.mouse_pos)

    def mouse_button1_release(self, event):
        self.button1_isdown = False
        self.get_mouse_pos(event)
        self.segment_windows[self.current_frame_n].release_mouse(self.mouse_pos, label=self.class_selector.curselection()[0])
        self.set_segment()
        self.var_class_label.set(self.class_labels[self.segment_windows[self.current_frame_n].label])

    def mouse_motion(self, event):
        self.get_mouse_pos(event)

    def next_frame(self, event=None, frames=1):
        self.current_frame_n = (self.current_frame_n + frames) % self.total_frames
        self.update_frame()

    def clear_segment(self, event=None):
        self.segment_windows[self.current_frame_n].is_active = False
        self.segment_windows[self.current_frame_n].label = -1
        #self.set_segment(clear=True)
        self.update_frame()

    def init_class_selector(self, master, labels):
        self.class_selector = tk.Listbox(master, selectmode=tk.SINGLE)
        self.class_selector.pack()
        
        for i in range(len(labels)):
            self.class_selector.insert(tk.END, labels[i])

class SegmentWindow(object):
    #### Rectangle selecting a frame segment as           ####
    #### a training example for training a sliding window ####
    def __init__(self, frame_width=640, frame_height=480, segment_width=32, segment_height=64):
        self.is_active = False
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.segment_width = segment_width
        self.segment_height = segment_height

        self.mouse_start = (0,0)
        
        self.mouse_window = (0,0,1,1)
        self.selection_window = (0,0,1,1)

        self.segment_window = np.zeros(1)

        self.label = -1

    def press_mouse(self, pos):
        self.mouse_start = pos

    def release_mouse(self, pos, label=-1):
        self.is_active = True
        self.selection_window = self.get_mouse_window(pos)
        self.selection_window = self.resize_window(self.selection_window)
        self.label = label

    def get_mouse_window(self, pos):
        return (self.mouse_start[0], self.mouse_start[1],
                pos[0] - self.mouse_start[0], pos[1] - self.mouse_start[1])

    def get_resized_window(self, pos):
        return self.resize_window((self.mouse_start[0], self.mouse_start[1],
                pos[0] - self.mouse_start[0], pos[1] - self.mouse_start[1]))

    def get_segment(self, image):
        x,y,w,h = self.selection_window
        
        segment = image[x:x+w,y:y+h].transpose(1,0,2)

        return scipy.misc.imresize(segment, (self.segment_height,self.segment_width))

    def resize_window(self, window):
        x,y,w,h = window

        if w == 0:
            w = 1
        elif w < 0:
            x += w
            w = -w

        if h == 0:
            h = 1
        elif h < 0:
            y += h
            h = -h

        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > self.frame_width: w = self.frame_width - x
        if y + h > self.frame_height: h = self.frame_height - y

        h = int(self.segment_height / self.segment_width) * w

        if y + h > self.frame_height:
            y = self.frame_height - h
        
        return (x, y, w, h)

if __name__ == "__main__":
    NLabeler()
