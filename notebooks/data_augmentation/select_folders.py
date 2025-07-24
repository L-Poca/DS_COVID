# select_folders.py
from ipywidgets import Text, VBox, Button, Output
from IPython.display import display
import os

class FolderSelector:
    def __init__(self):
        self.chemins = []
        self.fini = False
        self.out = Output()

        self.input_box = Text(placeholder="Collez un chemin vers un dossier")
        self.bouton_valider = Button(description="Ajouter")
        self.bouton_terminer = Button(description="Terminer")

        self.bouton_valider.on_click(self.ajouter_dossier)
        self.bouton_terminer.on_click(self.terminer)

        self.ui = VBox([
            self.input_box,
            self.bouton_valider,
            self.bouton_terminer,
            self.out
        ])

    def afficher(self):
        display(self.ui)

    def ajouter_dossier(self, _):
        chemin = self.input_box.value.strip()
        if chemin == "":
            with self.out:
                print("⚠️ Chemin vide ignoré.")
            return
        if os.path.isdir(chemin):
            self.chemins.append(chemin)
            with self.out:
                print(f"✅ Ajouté : {chemin}")
        else:
            with self.out:
                print(f"❌ Invalide : {chemin}")
        self.input_box.value = ""

    def terminer(self, _):
        self.bouton_valider.disabled = True
        self.bouton_terminer.disabled = True
        self.input_box.disabled = True
        self.fini = True
        with self.out:
            print("\n✔️ Saisie terminée.")
            print("📁 Dossiers retenus :")
            for path in self.chemins:
                print(f" - {path}")
