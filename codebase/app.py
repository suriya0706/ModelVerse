from app_utils import add_label, add_entry, add_combobox, add_button
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from tkinter import ttk
import tkinter as tk
import threading
import torch


# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")
input_path, output_path = None, None
mode, choosen_model = None, None
image_size, epochs, batch_size = None, None, None
status_label, settings_frame, file_frame, action_frame = None, None, None, None
widgets = {}


# Utility Functions
def initialize_frames(root):
    """Create and return the main UI frames."""
    settings_frame = ttk.Frame(root, relief="groove", padding=10)
    settings_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

    file_frame = ttk.Frame(root, relief="groove", padding=10)
    file_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

    action_frame = ttk.Frame(root, relief="groove", padding=10)
    action_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

    return settings_frame, file_frame, action_frame


def create_status_bar(root):
    """Create and return the status bar."""
    status_label = ttk.Label(root, text="Ready", anchor="center", relief="sunken",
                             background="#00bcd4", foreground="white")
    status_label.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
    return status_label


def create_mode_selection(root, mode, change_mode_callback):
    """Create and return the mode selection radio buttons."""
    mode_frame = ttk.Frame(root, padding=10, relief="groove")
    mode_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

    mode_label = ttk.Label(mode_frame, text="Mode:", font=("Helvetica Neue", 14), foreground="#333333")
    mode_label.grid(row=0, column=0, padx=10, pady=5)

    radio_button_train = ttk.Radiobutton(
        mode_frame, text="Train", variable=mode, value="train", command=change_mode_callback)
    radio_button_train.grid(row=0, column=1, padx=10, pady=5, sticky="w")

    radio_button_detect = ttk.Radiobutton(
        mode_frame, text="Detect", variable=mode, value="detect", command=change_mode_callback)
    radio_button_detect.grid(row=0, column=2, padx=10, pady=5, sticky="w")

    return mode_frame, mode_label, radio_button_train, radio_button_detect

def set_train_widgets():
    """Configure widgets for training mode and store them in the widgets dictionary."""
    widgets['experiment_label'] = add_label(settings_frame, "Experiment Name:", row=0, column=0)
    widgets['experiment_entry'] = add_entry(settings_frame, row=0, column=1)

    widgets['model_label'] = add_label(settings_frame, "Choose Model:", row=1, column=0)
    widgets['model_combobox'] = add_combobox(
        settings_frame, row=1, column=1, values=["yolo7", "yolo11"], default="yolo11", variable=choosen_model)

    widgets['input_button'] = add_button(file_frame, "Input (YAML file)", browse_input_folder, row=0, column=0)
    widgets['output_button'] = add_button(file_frame, "Output Folder", browse_output_folder, row=0, column=1)

    widgets['img_size_label'] = add_label(file_frame, "Image Size:", row=1, column=0)
    widgets['img_size_combobox'] = add_combobox(
        file_frame, row=1, column=1, values=[416, 512, 640, 1024, 1280], default=640, variable=image_size)

    widgets['epochs_label'] = add_label(file_frame, "Epochs:", row=2, column=0)
    widgets['epochs_combobox'] = add_combobox(
        file_frame, row=2, column=1, values=[1, 5, 10, 50, 100], default=100, variable=epochs)

    widgets['batch_label'] = add_label(file_frame, "Batch Size:", row=3, column=0)
    widgets['batch_combobox'] = add_combobox(
        file_frame, row=3, column=1, values=[8, 16, 32, 64], default=16, variable=batch_size)

    widgets['train_button'] = add_button(action_frame, "Train", run_model, row=0, column=0)


def set_detect_widgets():
    """Configure widgets for detection mode and store them in the widgets dictionary."""
    widgets['experiment_label'] = add_label(settings_frame, "Experiment Name:", row=0, column=0)
    widgets['experiment_entry'] = add_entry(settings_frame, row=0, column=1)

    widgets['model_label'] = add_label(settings_frame, "Choose Model:", row=1, column=0)
    widgets['model_combobox'] = add_combobox(
        settings_frame, row=1, column=1, values=["yolo7", "yolo11"], default="yolo11", variable=choosen_model)

    widgets['input_button'] = add_button(file_frame, "Input Folder", browse_input_folder, row=0, column=0)
    widgets['output_button'] = add_button(file_frame, "Output Folder", browse_output_folder, row=0, column=1)

    widgets['img_size_label'] = add_label(file_frame, "Image Size:", row=1, column=0)
    widgets['img_size_combobox'] = add_combobox(
        file_frame, row=1, column=1, values=[416, 512, 640, 1024, 1280], default=640, variable=image_size)

    widgets['detect_button'] = add_button(action_frame, "Detect", run_model, row=0, column=0)


def browse_input_folder():
    """Open a file dialog to select input path."""
    global input_path
    if mode.get() == "train":
        input_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])
        return input_path
    elif mode.get() == "detect":
        input_path = filedialog.askdirectory(title="Select Input Folder")
        return input_path


def browse_output_folder():
    """Open a file dialog to select output path."""
    global output_path
    output_path = filedialog.askdirectory(title="Select Output Folder")
    return output_path


def change_mode():
    """Switch between Train and Detect modes and update widgets."""
    for widget in settings_frame.winfo_children():
        widget.grid_forget()
    for widget in file_frame.winfo_children():
        widget.grid_forget()
    for widget in action_frame.winfo_children():
        widget.grid_forget()

    if mode.get() == "train":
        set_train_widgets()
    else:
        set_detect_widgets()


def run_model():
    """Run the YOLO model based on the selected mode."""
    if mode.get() == "train":
        print("Training the model...")
        model.train(data=input_path, imgsz=image_size.get(), epochs=epochs.get(), batch=batch_size.get(), project=output_path, name=widgets["experiment_entry"].get())
    else:
        print("Detecting with the model...")
        model.predict(source=input_path, imgsz=image_size.get(), project=output_path, name=widgets["experiment_entry"].get(), save=True)


def setup_gui(root):
    """Set up the main GUI."""
    global mode, choosen_model, image_size, epochs, batch_size, status_label, settings_frame, file_frame, action_frame

    # Initialize Tkinter variables
    mode = tk.StringVar(value="train")
    choosen_model = tk.StringVar(value="yolo11")
    image_size = tk.IntVar(value=640)
    epochs = tk.IntVar(value=100)
    batch_size = tk.IntVar(value=16)

    # Configure root window
    root.title("YOLO Training and Detection")
    root.geometry("450x600")
    root.configure(bg="#f5f5f5")
    root.resizable(True, True)

    # Apply styles
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TLabel", font=("Helvetica Neue", 12), padding=5, background="#f5f5f5", foreground="#333333")
    style.configure("TButton", font=("Helvetica Neue", 12), padding=6, relief="flat", background="#00bcd4", foreground="white")
    style.map("TButton", background=[("active", "#80e0e0")])
    style.configure("TRadiobutton", font=("Helvetica Neue", 12), padding=5, background="#f5f5f5", foreground="#333333")

    # Initialize frames
    settings_frame, file_frame, action_frame = initialize_frames(root)

    # Create status bar and mode selection
    status_label = create_status_bar(root)
    create_mode_selection(root, mode, change_mode)

    # Setup initial widgets for the mode
    change_mode()


# Main function
if __name__ == "__main__":
    root = tk.Tk()
    setup_gui(root)
    root.mainloop()
