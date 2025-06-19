import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from model.metadata_loader import load_metadata, parse_size
from model.image_analyzer import collect_image_stats
from view.gui import GUI

def lancer_analyse():
    path = app.get_dossier_path()
    if not path:
        messagebox.showerror("Erreur", "Veuillez sélectionner le dossier racine.")
        return

    app.btn_analyse.config(state="disabled")
    app.label_status.config(text="⏳ Analyse en cours...")

    def run():
        output = []
        logs = []

        dossiers = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]

        app.progress_bar["maximum"] = len(dossiers)
        app.progress_bar["value"] = 0

        for i, dossier in enumerate(dossiers):
            dossier_path = os.path.join(path, dossier)

            for sous_dossier in os.listdir(dossier_path):
                sous_path = os.path.join(dossier_path, sous_dossier)
                if not os.path.isdir(sous_path):
                    continue

                # 📸 Analyse des images via model.image_analyzer
                img_stats = collect_image_stats(sous_path)

                # 🧾 Chargement des métadonnées via model.metadata_loader
                metadata_file = os.path.join(path, f"{dossier}.metadata.xlsx")
                correspondance_rate = "N/A"
                taille_check = "N/A"

                if os.path.exists(metadata_file):
                    meta_names, meta_sizes = load_metadata(metadata_file)
                    verifiables = meta_names & img_stats["names"]

                    if meta_names:
                        taux_non_corres = 100 - (len(verifiables) / len(meta_names) * 100)
                        correspondance_rate = f"{taux_non_corres:.2f}%"

                    if not verifiables:
                        taille_check = "N/A"
                    else:
                        taille_check = "OK"
                        for name in verifiables:
                            true_size = img_stats["dimensions"].get(name)
                            expected_size = meta_sizes.get(name)
                            if true_size != expected_size:
                                taille_check = "NOK"
                                logs.append(
                                    f"Taille non conforme : {name}.png - {true_size} vs {expected_size}"
                                )
                else:
                    logs.append(f"Aucun metadata trouvé pour {dossier}")

                moyenne_ko = img_stats["total_size"] / len(img_stats["names"]) / 1024 if img_stats["names"] else 0

                output.append({
                    "Nom du dossier": dossier,
                    "Nom du sous dossier": sous_dossier,
                    "Description": "",
                    "Disponibilité de la variable a priori": "Oui",
                    "Type informatique": ", ".join(img_stats["formats"]),
                    "Taux de non correspondance entre metadata et dossier": correspondance_rate,
                    "Gestion du taux de non correspondance": "",
                    "Distribution des valeurs": len(img_stats["names"]),
                    "Remarques sur la colonne": "",
                    "Taille totale (Mo)": round(img_stats["total_size"] / (1024 * 1024), 2),
                    "Nb fichiers": len(img_stats["names"]),
                    "Taille moyenne (Ko)": round(moyenne_ko, 2),
                    "Mode couleur": ", ".join(img_stats["color_modes"]),
                    "Taille d'image conforme au metadata": taille_check,
                    "Dimensions uniques trouvées": ", ".join(str(s) for s in img_stats["shapes"]),
                })

            app.progress_bar["value"] = i + 1
            app.label_status.config(text=f"📦 Traitement {i + 1} / {len(dossiers)}")
            app.root.update_idletasks()

        now = datetime.now().strftime("%d_%m_%Y_%H_%M")
        os.makedirs("reports/explorationdata", exist_ok=True)
        csv_path = f"reports/explorationdata/Template_rapport_exploration_donnees_{now}.csv"
        pd.DataFrame(output).to_csv(csv_path, index=False, encoding="utf-8-sig")

        messagebox.showinfo("Analyse terminée", f"✅ Rapport généré :\n{csv_path}")
        app.label_status.config(text="✅ Analyse terminée.")
        app.btn_analyse.config(state="normal")

        if logs:
            log_path = f"reports/explorationdata/logs_{now}.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in logs)
            messagebox.showwarning("Anomalies détectées", f"{len(logs)} anomalies enregistrées dans {log_path}")

    threading.Thread(target=run).start()


# Lancement
root = tk.Tk()
app = GUI(root, lancer_analyse)
root.mainloop()