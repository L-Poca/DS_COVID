import os
from model.metadata_loader import load_metadata
from model.image_analyzer import collect_image_stats


def run_analysis(path):
    output = []
    logs = []

    for dossier in get_dossiers(path):
        dossier_path = os.path.join(path, dossier)

        for sous_dossier, sous_path in get_sous_dossiers(dossier_path):
            img_stats = collect_image_stats(sous_path)
            metadata_file = os.path.join(path, f"{dossier}.metadata.xlsx")

            correspondance_rate, taille_check, log_entries = analyze_metadata(
                metadata_file, img_stats
            )
            logs.extend(log_entries)

            output.append(build_output_entry(
                dossier, sous_dossier, img_stats,
                correspondance_rate, taille_check
            ))

    return output, logs


def get_dossiers(path):
    return [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]


def get_sous_dossiers(dossier_path):
    for sous_dossier in os.listdir(dossier_path):
        sous_path = os.path.join(dossier_path, sous_dossier)
        if os.path.isdir(sous_path):
            yield sous_dossier, sous_path


def analyze_metadata(metadata_file, img_stats):
    logs = []
    correspondance_rate = "N/A"
    taille_check = "N/A"

    if not os.path.exists(metadata_file):
        logs.append(
            f"Aucun metadata trouvé pour {os.path.basename(metadata_file)}"
            )
        return correspondance_rate, taille_check, logs

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
                    f"Taille non conforme : {name}.png -"
                    f" {true_size} vs {expected_size}"
                )

    return correspondance_rate, taille_check, logs


def build_output_entry(
        dossier,
        sous_dossier,
        img_stats,
        correspondance_rate,
        taille_check
        ):
    moyenne_ko = (
        img_stats["total_size"] / len(img_stats["names"]) / 1024
        if img_stats["names"] else 0
    )

    return {
        "Nom du dossier": dossier,
        "Nom du sous dossier": sous_dossier,
        "Description": "",
        "Disponibilité de la variable a priori": "Oui",
        "Type informatique": ", ".join(img_stats["formats"]),
        "Taux de non correspondance entre metadata et dossier": correspondance_rate, # noqa
        "Gestion du taux de non correspondance": "",
        "Distribution des valeurs": len(img_stats["names"]),
        "Remarques sur la colonne": "",
        "Taille totale (Mo)": round(
            img_stats["total_size"] / (1024 * 1024), 2
            ),
        "Nb fichiers": len(img_stats["names"]),
        "Taille moyenne (Ko)": round(moyenne_ko, 2),
        "Mode couleur": ", ".join(img_stats["color_modes"]),
        "Taille d'image conforme au metadata": taille_check,
        "Dimensions uniques trouvées": ", ".join(
            str(s) for s in img_stats["shapes"]
            ),
    }
