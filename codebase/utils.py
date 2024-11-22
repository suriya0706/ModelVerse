from tkinter import filedialog
from ultralytics import YOLO

# Global variables
input_path, output_path, model = None, None, YOLO("/path/to/your/model.pt")


def browse_input_folder():
    """Open a file dialog to select input path."""
    global input_path
    input_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])


def browse_output_folder():
    """Open a file dialog to select output path."""
    global output_path
    output_path = filedialog.askdirectory(title="Select Output Folder")


def run_model():
    """Run the YOLO model based on the selected mode."""
    if mode.get() == "train":
        print("Training the model...")
        model.train(data=input_path, imgsz=image_size.get(), epochs=epochs.get(), batch=batch_size.get())
    else:
        print("Detecting with the model...")
        model.predict(source=input_path, save=True, imgsz=image_size.get())
