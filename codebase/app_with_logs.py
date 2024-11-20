import threading
from tkinter import filedialog, messagebox
from tkinter import ttk
import tkinter as tk
from ultralytics import YOLO
import torch
import sys
import io

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training and Detection")
        self.root.geometry("800x600")  # Default size
        self.root.resizable(True, True)

        # Grid layout configuration for dynamic resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=0)  # Status row
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)

        # Define styles for a modern look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Arial", 12), padding=5, background="#f7f7f7")
        style.configure("TButton", font=("Arial", 12), padding=6, relief="flat", background="#000000", foreground="white")
        style.map("TButton", background=[("active", "#11302a")])
        style.configure("TRadiobutton", font=("Arial", 12), padding=5, background="#f7f7f7")
        style.configure("TEntry", font=("Arial", 12), padding=5, relief="flat")
        style.configure("TCombobox", font=("Arial", 12), padding=5, relief="flat")
        style.configure("TFrame", background="#f7f7f7")

        # Device and model initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")
        self.input_path = None
        self.output_path = None
        self.mode = tk.StringVar()
        self.mode.set("train")
        self.choosen_model = tk.StringVar()
        self.choosen_model.set("yolo11")
        self.image_size = tk.IntVar()
        self.image_size.set(640)  # Default image size
        self.epochs = tk.IntVar()
        self.epochs.set(100)  # Default epochs
        self.batch_size = tk.IntVar()
        self.batch_size.set(16)  # Default batch size

        # Main layout frames
        self.mode_frame = ttk.Frame(root, padding=10, relief="groove")
        self.mode_frame.grid(row=0, column=0, padx=10, sticky="ew")

        self.settings_frame = ttk.Frame(root, relief="groove")
        self.settings_frame.grid(row=1, column=0, padx=10, sticky="ew")

        self.file_frame = ttk.Frame(root, relief="groove")
        self.file_frame.grid(row=2, column=0, padx=10, sticky="ew")

        self.action_frame = ttk.Frame(root, relief="groove")
        self.action_frame.grid(row=3, column=0, padx=10, sticky="ew")

        # Status label
        self.status_label = ttk.Label(root, text="Ready", anchor="center", relief="sunken", background="#f7f7f7")
        self.status_label.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        # Mode selection radio buttons
        ttk.Label(self.mode_frame, text="Mode:").grid(row=0, column=0, padx=10, pady=5)
        self.radio_button_train = ttk.Radiobutton(self.mode_frame, text="Train", variable=self.mode, value="train", command=self.change_mode)
        self.radio_button_train.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.radio_button_detect = ttk.Radiobutton(self.mode_frame, text="Detect", variable=self.mode, value="detect", command=self.change_mode)
        self.radio_button_detect.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        # Logs frame with scrolling
        self.logs_frame = ttk.Frame(root, padding=10, relief="groove")
        self.logs_frame.grid(row=0, column=1, rowspan=5, padx=10, pady=10, sticky="nsew")

        # Scrollbar setup
        self.canvas = tk.Canvas(self.logs_frame)
        self.scrollbar = ttk.Scrollbar(self.logs_frame, orient="vertical", command=self.canvas.yview)
        self.log_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.create_window((0, 0), window=self.log_frame, anchor="nw")

        self.log_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Set initial widgets
        self.dynamic_widgets = []
        self.change_mode()

    def set_train_widgets(self):
        self.add_experiment_widgets(self.settings_frame, row=0, column=0)
        self.add_model_selection(self.settings_frame, row=1, column=0)
        self.add_file_selection(self.file_frame, ".yaml file", "Output folder", row=0, column=0)
        self.add_image_size(self.file_frame, row=1, column=0)
        self.add_epochs_and_batch(self.file_frame, row=2, column=0)
        self.add_run_button(self.action_frame, "Train", row=5, column=0)

    def set_detect_widgets(self):
        self.add_experiment_widgets(self.settings_frame, row=0, column=0)
        self.add_model_selection(self.settings_frame, row=1, column=0)
        self.add_file_selection(self.file_frame, "Input folder", "Output folder", row=2, column=0)
        self.add_image_size(self.file_frame, row=3, column=0)
        self.add_run_button(self.action_frame, "Detect", row=4, column=0)

    def add_experiment_widgets(self, frame, row, column):
        self.add_label(frame, "Experiment Name:", row=row, column=column)
        self.experiment_name_entry = self.add_entry(frame, row=row, column=column+1)
        self.dynamic_widgets.append(self.experiment_name_entry)

    def add_model_selection(self, frame, row, column):
        self.add_label(frame, "Choose Model:", row=row, column=column)

        # Create the combobox for model selection
        self.model_combobox = self.add_combobox(frame, row=row, column=column+1, values=["yolo7", "yolo11"], default="yolo11", variable=self.choosen_model)

        # Update model on selection change
        self.model_combobox.bind("<<ComboboxSelected>>", self.change_mode)  # Corrected here
        self.dynamic_widgets.append(self.model_combobox)

    def add_file_selection(self, frame, input_text, output_text, row, column):
        self.input_button = self.add_button(frame, input_text, self.browse_input_folder, row=row, column=column)
        self.output_button = self.add_button(frame, output_text, self.browse_output_folder, row=row, column=column+1)
        self.dynamic_widgets.extend([self.input_button, self.output_button])

    def add_image_size(self, frame, row, column):
        self.add_label(frame, "Image Size:", row=row, column=column)
        self.image_size_combobox = self.add_combobox(frame, row=row, column=column+1, values=[416, 512, 640, 1024, 1280], default=640, variable=self.image_size)
        self.dynamic_widgets.append(self.image_size_combobox)

    def add_epochs_and_batch(self, frame, row, column):
        # Epochs
        self.add_label(frame, "Epochs:", row=row, column=column)
        self.epochs_combobox = self.add_combobox(frame, row=row, column=column+1, values=[1, 5, 10, 50, 100, 200, 300], default=100, variable=self.epochs)
        self.dynamic_widgets.append(self.epochs_combobox)

        # Batch Size
        self.add_label(frame, "Batch Size:", row=row+1, column=column)
        self.batch_combobox = self.add_combobox(frame, row=row+1, column=column+1, values=[2, 4, 8, 16, 32], default=16, variable=self.batch_size)
        self.dynamic_widgets.append(self.batch_combobox)

    def add_run_button(self, frame, text, row, column):
        self.run_button = self.add_button(frame, text=text, command=self.run_model, row=row, column=column)
        self.dynamic_widgets.append(self.run_button)

    def add_label(self, frame, text, row, column, sticky="w"):
        label = ttk.Label(frame, text=text)
        label.grid(row=row, column=column, padx=10, pady=10, sticky=sticky)
        return label

    def add_button(self, frame, text, command, row, column):
        button = ttk.Button(frame, text=text, command=command)
        button.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
        return button

    def add_combobox(self, frame, row, column, values, default, variable=None):
        combobox = ttk.Combobox(frame, values=values, state="readonly")
        combobox.set(default)
        combobox.grid(row=row, column=column, padx=10, pady=10, sticky="ew")

        if variable:
            combobox.bind("<<ComboboxSelected>>", lambda event: variable.set(combobox.get()))  # Update the variable on selection change

        return combobox

    def add_entry(self, frame, row, column):
        entry = ttk.Entry(frame)
        entry.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
        return entry

    def change_mode(self):
        mode = self.mode.get()
        for widget in self.dynamic_widgets:
            widget.grid_forget()  # Hide all dynamic widgets

        if mode == "train":
            self.set_train_widgets()
        elif mode == "detect":
            self.set_detect_widgets()

    def browse_input_folder(self):
        self.input_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])
        print(f"Selected input folder: {self.input_path}")

    def browse_output_folder(self):
        self.output_path = filedialog.askdirectory(title="Select Output Folder")
        print(f"Selected output folder: {self.output_path}")

    def run_model(self):
        self.status_label.config(text="Running Model...")
        self.root.update()

        # Add code for model training or detection logic here
        if self.mode.get() == "train":
            self.train_model()
        elif self.mode.get() == "detect":
            self.detect_model()

    def train_model(self):
        experiment_name = self.experiment_name_entry.get()
        image_size = self.image_size.get()
        epochs = self.epochs.get()
        batch_size = self.batch_size.get()

        # Placeholder for training logic
        print(f"Training with Image Size: {image_size}, Epochs: {epochs}, Batch Size: {batch_size}")
        self.model.train(data=self.input_path, epochs=epochs, batch=batch_size, imgsz=image_size, project=self.output_path, name=experiment_name, device=self.device)
        self.status_label.config(text="Training Complete")

    def detect_model(self):
        experiment_name = self.experiment_name_entry.get()
        image_size = self.image_size.get()

        # Perform detection
        print(f"Detecting with Model: {self.choosen_model.get()} and Image Size: {image_size}")
        self.model.predict(source=self.input_path, imgsz=image_size, save=True, project=self.output_path, device=self.device)
        self.status_label.config(text="Detection Complete")


# Running the app
root = tk.Tk()
app = App(root)
root.mainloop()
