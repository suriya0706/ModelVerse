from tkinter import filedialog, messagebox
from tkinter import ttk
import tkinter as tk
from ultralytics import YOLO
import torch

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training and Detection")
        self.root.geometry("800x600")  # Default size # Set a minimum size for the window
        self.root.resizable(True, True)

        # Grid layout configuration for dynamic resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=0) 
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)

        # Define styles for a modern look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Arial", 12), padding=5, background="#f7f7f7")
        style.configure("TButton", font=("Arial", 12), padding=6, relief="flat", background="#000000", foreground="white")
        style.map("TButton", background=[("active", "#")])
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
        self.image_size = tk.StringVar()
        self.image_size.set("Select Image size")

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

        # Mode selection
        ttk.Label(self.mode_frame, text="Mode:").grid(row=0, column=0, padx=10, pady=5)
        self.radio_button_train = ttk.Radiobutton(self.mode_frame, text="Train", variable=self.mode, value="train", command=self.change_mode)
        self.radio_button_train.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.radio_button_detect = ttk.Radiobutton(self.mode_frame, text="Detect", variable=self.mode, value="detect", command=self.change_mode)
        self.radio_button_detect.grid(row=0, column=2, padx=10, pady=5, sticky="w")

        self.logs_frame = ttk.Frame(root, padding=10, relief="groove")
        self.logs_frame.grid(row=0, column=1, rowspan=5, padx=10, pady=10, sticky="nsew")
        self.logs_frame.grid_rowconfigure(0, weight=1)  
        self.logs_frame.grid_columnconfigure(0, weight=1) 
        self.label_logs_frame = ttk.Label(self.logs_frame, text="Logs will appear here", anchor="nw", justify="left", wraplength=200)
        self.label_logs_frame.grid(row=0, column=0, sticky="nsew")


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
        self.model_combobox = self.add_combobox(frame, row=row, column=column+1, values=["yolo7", "yolo11"], default="yolo11")
        
        # Update model on selection change
        self.model_combobox.bind("<<ComboboxSelected>>", self.change_model)
        self.dynamic_widgets.append(self.model_combobox)

    def add_file_selection(self, frame, input_text, output_text, row, column):
        self.input_button = self.add_button(frame, input_text, self.browse_input_folder, row=row, column=column)
        self.output_button = self.add_button(frame, output_text, self.browse_output_folder, row=row, column=column+1)
        self.dynamic_widgets.extend([self.input_button, self.output_button])

    def add_image_size(self, frame, row, column):
        self.add_label(frame, "Image Size:", row=row, column=column)
        self.image_size_combobox = self.add_combobox(frame, row=row, column=column+1, values=[412, 512, 640, 1024, 1280], default=640)
        self.dynamic_widgets.append(self.image_size_combobox)

    def add_epochs_and_batch(self, frame, row, column):
        # Epochs
        self.add_label(frame, "Epochs:", row=row, column=column)
        self.epochs_combobox = self.add_combobox(frame, row=row, column=column+1, values=[1, 5, 10, 50, 100, 200, 300], default=100)
        self.dynamic_widgets.append(self.epochs_combobox)

        # Batch Size
        self.add_label(frame, "Batch Size:", row=row+1, column=column)
        self.batch_combobox = self.add_combobox(frame, row=row+1, column=column+1, values=[2, 4, 8, 16, 32], default=16)
        self.dynamic_widgets.append(self.batch_combobox)

    def add_run_button(self, frame, text, row, column):
        self.run_button = self.add_button(frame, text=text, command=self.run_model, row=row, column=column)
        self.dynamic_widgets.append(self.run_button)

    def add_label(self, frame, text, row, column, sticky="w"):
        label = ttk.Label(frame, text=text)
        label.grid(row=row, column=column, padx=10, pady=10, sticky=sticky)
        self.dynamic_widgets.append(label)

    def add_entry(self, frame, row, column):
        entry = ttk.Entry(frame)
        entry.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
        return entry

    def add_combobox(self, frame, row, column, values, default=None):
        combobox = ttk.Combobox(frame, values=values, state="readonly")
        if default is not None:
            combobox.set(default)
        combobox.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
        return combobox

    def add_button(self, frame, text, command, row, column):
        button = ttk.Button(frame, text=text, command=command)
        button.grid(row=row, column=column, padx=10, pady=10, sticky="ew")
        return button

    def change_mode(self):
        for widget in self.dynamic_widgets:
            widget.grid_forget()
        self.dynamic_widgets.clear()
        if self.mode.get() == "train":
            self.set_train_widgets()
        elif self.mode.get() == "detect":
            self.set_detect_widgets()

    def change_model(self, event=None):
        selected_model = self.model_combobox.get()
        if selected_model == "yolo11":
            self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")
        elif selected_model == "yolo7":
            self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo7.pt")

    def browse_input_folder(self):
        self.input_path = filedialog.askdirectory()
        if self.input_path:
            print(f"Selected input folder: {self.input_path}")

    def browse_output_folder(self):
        self.output_path = filedialog.askdirectory()
        if self.output_path:
            print(f"Selected output folder: {self.output_path}")

    def run_model(self):
        experiment_name = self.experiment_name_entry.get()
        image_size = self.image_size_combobox.get()
        epochs = self.epochs_combobox.get()
        batch_size = self.batch_combobox.get()
        # Add logic to train or detect based on the mode and inputs
        print(f"Running {self.mode.get()} with {experiment_name}, Image Size: {image_size}, Epochs: {epochs}, Batch Size: {batch_size}")
        messagebox.showinfo("Process", f"Running {self.mode.get()} with selected parameters.")



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
