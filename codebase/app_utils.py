import threading
from tkinter import filedialog, messagebox
from tkinter import ttk
import tkinter as tk


def add_label(frame, text, row, column):
    """Add a label to the frame and return it."""
    label = ttk.Label(frame, text=text, font=("Helvetica Neue", 12), foreground="#333333")
    label.grid(row=row, column=column, sticky="w", padx=10, pady=5)
    return label


def add_entry(frame, row, column):
    """Add an entry box to the frame and return it."""
    entry = ttk.Entry(frame, font=("Helvetica Neue", 12), foreground="#333333", background="#ffffff")
    entry.grid(row=row, column=column, padx=10, pady=5)
    return entry


def add_combobox(frame, row, column, values, default, variable):
    """Add a combobox to the frame and return it."""
    combobox = ttk.Combobox(frame, values=values, state="readonly", textvariable=variable, font=("Helvetica Neue", 12))
    combobox.set(default)
    combobox.grid(row=row, column=column, padx=10, pady=5)
    return combobox


def add_button(frame, text, command, row, column):
    """Add a button to the frame and return it."""
    button = ttk.Button(frame, text=text, command=command, style="TButton")
    button.grid(row=row, column=column, padx=10, pady=5)
    return button