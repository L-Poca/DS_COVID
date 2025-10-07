import tkinter as tk
from tkinter import filedialog, ttk


class GUI:
    def __init__(self, root, on_analyse_callback):
        self.root = root
        self.root.title("Analyse automatique des images + metadata")

        tk.Label(
            root,
            text="📁 Dossier principal :"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=10)

        self.entry_root = tk.Entry(root, width=60)
        self.entry_root.grid(row=0, column=1, padx=5)

        tk.Button(
            root, text="Parcourir", command=self.choisir_dossier
        ).grid(row=0, column=2)

        self.btn_analyse = tk.Button(
            root, text="🚀 Lancer l’analyse",
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
            command=on_analyse_callback
        )
        self.btn_analyse.grid(row=1, column=1, pady=10)

        self.progress_bar = ttk.Progressbar(
            root,
            orient="horizontal",
            length=400,
            mode="determinate"
            )
        self.progress_bar.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

        self.label_status = tk.Label(root, text="", fg="gray")
        self.label_status.grid(row=3, column=0, columnspan=3, pady=5)

    def choisir_dossier(self):
        path = filedialog.askdirectory(title="Choisir le dossier principal")
        if path:
            self.entry_root.delete(0, tk.END)
            self.entry_root.insert(0, path)

    def get_dossier_path(self):
        return self.entry_root.get()
