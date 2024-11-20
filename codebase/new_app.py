from tkinter import filedialog, messagebox
from ultralytics import YOLO
import tkinter as tk
import torch

class App:
	def __init__(self, root):
		self.root = root
		self.root.title("YOLO Training and Detection")
		self.root.geometry("400x70")
		# self.root.configure(bg="#000000") 
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")
		self.input_path = None
		self.output_path = None
		self.mode=tk.StringVar()
		self.mode.set("train")
		self.radio_button_train = tk.Radiobutton(root, text='Train', variable=self.mode, value='train', command=self.change_mode, font=('Arial', 12))
		self.radio_button_train.place(x=20, y=25)
		self.radio_button_detect = tk.Radiobutton(root, text='Detect', variable=self.mode, value='detect', command=self.change_mode, font=('Arial', 12))
		self.radio_button_detect.place(x=100, y=25)
		self.choosen_model=tk.StringVar()
		self.choosen_model.set("yolo11")
		self.image_size=tk.StringVar()
		self.image_size.set("Select Image size")
		self.dynamic_widgets = []

	def set_train_widgets(self, root):
		self.experiment_name_label = tk.Label(root, text="Enter experiment name", font=('Arial', 12))
		self.experiment_name_label.place(x=20, y=75)
		self.experiment_name_entry = tk.Entry(root)
		self.experiment_name_entry.place(x=200, y=75)
		self.dynamic_widgets.extend([self.experiment_name_label, self.experiment_name_entry])

		self.choose_model_label=tk.Label(root, text="Choose model", font=('Arial', 12))
		self.choose_model_label.place(x=20, y=125)
		self.radio_button_yolov7 = tk.Radiobutton(root, text='Yolov7', variable=self.choosen_model, value='yolo7', command=self.change_model, font=('Arial', 12))
		self.radio_button_yolov7.place(x=140, y=125)
		self.radio_button_yolov11 = tk.Radiobutton(root, text='Yolov11', variable=self.choosen_model, value='yolo11', command=self.change_model, font=('Arial', 12))
		self.radio_button_yolov11.place(x=225, y=125)
		self.dynamic_widgets.extend([self.choose_model_label, self.radio_button_yolov7, self.radio_button_yolov11])

		self.input_browse_button = tk.Button(root, text="Select .yaml file for training", command=self.browse_input_folder, font=('Arial', 12), width=55)
		self.input_browse_button.place(x=20, y=175)
		self.output_browse_button = tk.Button(root, text="Select output folder to save logs and model weights", command=self.browse_output_folder, font=('Arial', 12), width=55)
		self.output_browse_button.place(x=20, y=225)	
		self.dynamic_widgets.extend([self.input_browse_button, self.output_browse_button])

		self.weight_browse_button = tk.Button(root, text="Select custom model weight", command=self.browse_weights, font=('Arial', 12))
		self.weight_browse_button.place(x=20, y=275)
		self.dynamic_widgets.extend([self.weight_browse_button])

		self.image_size_dropbox = tk.OptionMenu(root, self.image_size, 412, 512, 640, 1024, 1280)
		self.image_size_dropbox.place(x=280, y=275)
		self.image_size_label = tk.Label(root, text="Image size", font=('Arial', 12))
		self.image_size_label.place(x=280, y=300)
		self.dynamic_widgets.extend([self.image_size_dropbox, self.image_size_label])

		self.epochs_slider = tk.Scale(root, from_=1, to=300, orient="horizontal", length=100, resolution=1, font=('Arial', 12))
		self.epochs_slider.set(100)
		self.epochs_slider.place(x=20, y=325)
		self.epoch_label=tk.Label(root, text="Epochs", font=('Arial', 12))
		self.epoch_label.place(x=20, y=365)
		self.dynamic_widgets.extend([self.epochs_slider, self.epoch_label])

		self.batch_slider = tk.Scale(root, from_=2, to=32, orient="horizontal", length=100, resolution=1, font=('Arial', 12))
		self.batch_slider.set(100)
		self.batch_slider.place(x=150, y=325)
		self.batch_label = tk.Label(root, text="Batch size", font=('Arial', 12))
		self.batch_label.place(x=150, y=365)
		self.dynamic_widgets.extend([self.batch_slider, self.batch_label])			
		
		self.run_button = tk.Button(root, text="Train", command=self.run_model, font=('Arial', 12), width=10)
		self.run_button.place(x=20, y=400)
		self.dynamic_widgets.extend([self.run_button])	
		self.root.geometry("505x460")

	def set_detect_widgets(self, root):
		self.experiment_name_label = tk.Label(root, text="Enter experiment name", font=('Arial', 12))
		self.experiment_name_label.place(x=20, y=75)
		self.experiment_name_entry = tk.Entry(root)
		self.experiment_name_entry.place(x=200, y=75)
		self.dynamic_widgets.extend([self.experiment_name_label, self.experiment_name_entry])

		self.choose_model_label=tk.Label(root, text="Choose model", font=('Arial', 12))
		self.choose_model_label.place(x=20, y=125)
		self.radio_button_yolov7 = tk.Radiobutton(root, text='Yolov7', variable=self.choosen_model, value='yolo7', command=self.change_model, font=('Arial', 12))
		self.radio_button_yolov7.place(x=140, y=125)
		self.radio_button_yolov11 = tk.Radiobutton(root, text='Yolov11', variable=self.choosen_model, value='yolo11', command=self.change_model, font=('Arial', 12))
		self.radio_button_yolov11.place(x=225, y=125)
		self.dynamic_widgets.extend([self.choose_model_label, self.radio_button_yolov7, self.radio_button_yolov11])

		self.input_browse_button = tk.Button(root, text="Select input folder containing input images", command=self.browse_input_folder, font=('Arial', 12), width=55)
		self.input_browse_button.place(x=20, y=175)
		self.output_browse_button = tk.Button(root, text="Select output folder to save logs and model weights", command=self.browse_output_folder, font=('Arial', 12), width=55)
		self.output_browse_button.place(x=20, y=225)	
		self.dynamic_widgets.extend([self.input_browse_button, self.output_browse_button])

		self.weight_browse_button = tk.Button(root, text="Select custom model weight", command=self.browse_weights, font=('Arial', 12))
		self.weight_browse_button.place(x=20, y=275)
		self.dynamic_widgets.extend([self.weight_browse_button])

		self.image_size_dropbox = tk.OptionMenu(root, self.image_size, 412, 512, 640, 1024, 1280)
		self.image_size_dropbox.place(x=280, y=275)
		self.image_size_label = tk.Label(root, text="Image size", font=('Arial', 12))
		self.image_size_label.place(x=280, y=300)
		self.dynamic_widgets.extend([self.image_size_dropbox, self.image_size_label])
		
		self.run_button = tk.Button(root, text="Detect", command=self.run_model, font=('Arial', 12), width=10)
		self.run_button.place(x=20, y=335)
		self.dynamic_widgets.extend([self.run_button])	
		self.root.geometry("505x395")

	def change_mode(self):
		for widget in self.dynamic_widgets:
			if widget:
				widget.destroy() 
		self.dynamic_widgets.clear()
		if self.mode.get() == "train":
			self.set_train_widgets(root)
		elif self.mode.get() == "detect":
			self.set_detect_widgets(root)
		self.root.update()

	def change_model(self):
		if self.choosen_model.get() == "yolo7":
			self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")
		elif self.choosen_model.get() == "yolo11":
			self.model = YOLO("/home/jarvis/Desktop/projects/FR/model_weights/yolo11n.pt")

	def browse_input_folder(self):
		if self.mode.get() == "train":
			self.input_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])
			print(self.input_path)
		elif self.mode.get() == "detect":
			self.input_path = filedialog.askdirectory(title="Select input folder containing input images")
			print(self.input_path)

	def browse_output_folder(self):
		if self.mode.get() == "train":
			self.output_path = filedialog.askdirectory(title="Select output folder to save logs and model weights")
			print(self.output_path)
		elif self.mode.get() == "detect":
			self.output_path = filedialog.askdirectory(title="Select output folder to save detections")
			print(self.output_path)

	def browse_weights(self):
		if self.mode.get() == "train":
			self.weight_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.pth *.pt")])
			print(self.weight_path)

	def run_model(self):
		if not self.input_path:
			messagebox.showerror("Error", "Please select a valid input file or folder")
			return
		elif not self.output_path:
			messagebox.showerror("Error", "Please select a valid output file or folder")
			return
		elif self.image_size.get() == "Select Image size":
			messagebox.showerror("Error", "Please select an image size.")
			return

		if self.mode.get() == "train":
			try:
				train_results = self.model.train(
					data=self.input_path,
					project=self.output_path,
					name=self.experiment_name_entry.get(),
					epochs=self.epochs_slider.get(),
					batch=self.batch_slider.get(),
					imgsz=self.image_size.get(),
					device=self.device
				)
				messagebox.showinfo("Success", "Training completed successfully!")
			except Exception as e:
				messagebox.showerror("Error", f"Training failed: {e}")
		elif self.mode.get() == "detect":
			try:
				train_results = self.model.predict(
					source=self.input_path,
					project=self.output_path,
					name=self.experiment_name_entry.get(),
					imgsz=self.image_size.get(),
					device=self.device,
					save=True,
					save_txt=True,
					save_conf=True
				)
				messagebox.showinfo("Success", "Detection completed successfully!")
			except Exception as e:
				messagebox.showerror("Error", f"Detection failed: {e}")

if __name__ == "__main__":
	root=tk.Tk()
	app=App(root)
	root.mainloop()
