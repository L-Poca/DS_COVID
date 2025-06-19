import pandas as pd


def parse_size(size_str):
    try:
        w, h = map(int, str(size_str).split("*"))
        return (w, h)
    except Exception:
        return None


def load_metadata(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df["parsed_size"] = df["SIZE"].apply(parse_size)
    return set(
        df["FILE NAME"].astype(str)
        ), dict(
            zip(df["FILE NAME"], df["parsed_size"])
            )
