import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from model.analysis_runner import run_analysis
from view.gui import GUI


def lancer_analyse():
    path = app.get_dossier_path()

    if not path:
        messagebox.showerror(
            "Erreur",
            "Veuillez sélectionner le dossier racine."
            )
        return

    app.btn_analyse.config(state="disabled")
    app.label_status.config(text="⏳ Analyse en cours...")

    def run():
        output, logs = run_analysis(path)

        # Mise à jour UI et export
        now = datetime.now().strftime("%d_%m_%Y_%H_%M")
        os.makedirs("reports/explorationdata", exist_ok=True)
        csv_path = (
            "reports/explorationdata/Template_rapport_exploration_donnees_"
            f"{now}.csv"
        )
        pd.DataFrame(output).to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig"
            )

        messagebox.showinfo(
            "Analyse terminée",
            f"✅ Rapport généré :\n{csv_path}"
            )
        app.label_status.config(text="✅ Analyse terminée.")
        app.btn_analyse.config(state="normal")

        if logs:
            log_path = f"reports/explorationdata/logs_{now}.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in logs)
            messagebox.showwarning(
                "Anomalies détectées",
                f"{len(logs)} anomalies enregistrées dans {log_path}"
                )

    # On lance le traitement dans un thread
    threading.Thread(target=run).start()


# Lancement de l'application
root = tk.Tk()
app = GUI(root, lancer_analyse)
root.mainloop()
