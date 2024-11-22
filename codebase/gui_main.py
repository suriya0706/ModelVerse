import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from utils import initialize_frames, create_status_bar, create_mode_selection, browse_input_folder, browse_output_folder, run_model

# Global variables
mode, choosen_model, image_size, epochs, batch_size = None, None, None, None, None
status_label, settings_frame, file_frame, action_frame = None, None, None, None
widgets = {}


def set_train_widgets():
    """Configure widgets for training mode and store them in the widgets dictionary."""
    widgets['experiment_label'] = initialize_frames.add_label(settings_frame, "Experiment Name:", row=0, column=0)
    widgets['experiment_entry'] = initialize_frames.add_entry(settings_frame, row=0, column=1)

    widgets['model_label'] = initialize_frames.add_label(settings_frame, "Choose Model:", row=1, column=0)
    widgets['model_combobox'] = initialize_frames.add_combobox(
        settings_frame, row=1, column=1, values=["yolo7", "yolo11"], default="yolo11", variable=choosen_model)

    widgets['input_button'] = initialize_frames.add_button(file_frame, "Input (YAML file)", browse_input_folder, row=0, column=0)
    widgets['output_button'] = initialize_frames.add_button(file_frame, "Output Folder", browse_output_folder, row=0, column=1)

    widgets['img_size_label'] = initialize_frames.add_label(file_frame, "Image Size:", row=1, column=0)
    widgets['img_size_combobox'] = initialize_frames.add_combobox(
        file_frame, row=1, column=1, values=[416, 512, 640, 1024, 1280], default=640, variable=image_size)

    widgets['epochs_label'] = initialize_frames.add_label(file_frame, "Epochs:", row=2, column=0)
    widgets['epochs_combobox'] = initialize_frames.add_combobox(
        file_frame, row=2, column=1, values=[1, 5, 10, 50, 100], default=100, variable=epochs)

    widgets['batch_label'] = initialize_frames.add_label(file_frame, "Batch Size:", row=3, column=0)
    widgets['batch_combobox'] = initialize_frames.add_combobox(
        file_frame, row=3, column=1, values=[8, 16, 32, 64], default=16, variable=batch_size)

    widgets['train_button'] = initialize_frames.add_button(action_frame, "Train", run_model, row=0, column=0)


def set_detect_widgets():
    """Configure widgets for detection mode and store them in the widgets dictionary."""
    widgets['experiment_label'] = initialize_frames.add_label(settings_frame, "Experiment Name:", row=0, column=0)
    widgets['experiment_entry'] = initialize_frames.add_entry(settings_frame, row=0, column=1)

    widgets['model_label'] = initialize_frames.add_label(settings_frame, "Choose Model:", row=1, column=0)
    widgets['model_combobox'] = initialize_frames.add_combobox(
        settings_frame, row=1, column=1, values=["yolo7", "yolo11"], default="yolo11", variable=choosen_model)

    widgets['input_button'] = initialize_frames.add_button(file_frame, "Input Folder", browse_input_folder, row=0, column=0)
    widgets['output_button'] = initialize_frames.add_button(file_frame, "Output Folder", browse_output_folder, row=0, column=1)

    widgets['img_size_label'] = initialize_frames.add_label(file_frame, "Image Size:", row=1, column=0)
    widgets['img_size_combobox'] = initialize_frames.add_combobox(
        file_frame, row=1, column=1, values=[416, 512, 640, 1024, 1280], default=640, variable=image_size)

    widgets['detect_button'] = initialize_frames.add_button(action_frame, "Detect", run_model, row=0, column=0)


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
