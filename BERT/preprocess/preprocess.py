import pandas as pd
import re
import unicodedata
import numpy as np


def get_expressions_to_delete():
    """
    Devuelve una lista de patrones *precompilados* para eliminar
    instrucciones o textos no deseados en el resumen.
    Los patrones están diseñados para tolerar saltos de línea y variaciones menores.
    """
    pats = [
        # "resumen del proyecto (1 página)"
         r"""
        ^[\s\u00A0]*                          # espacios normales o NBSP al inicio de línea
        (?:[IVXLCDM]+\.\s*|\.\s*)?            # opcional: 'IV.' / 'V.' ... o solo '.'
        resumen\s+del\s+proyecto              # texto base
        (?:\s*\(\s*1\s*p[aá]gina\s*\))?       # opcional: '(1 página)' o '(1PÁGINA)'
        """,


        #ii. objetivo general y objetivos específicos (1página) 
        r"""
        (?:ii\.\s*)?                               # "ii." opcional
        objetivo\s+general
        [\s\S]{0,100}?                             # margen de texto flexible
        objetivos?\s+espec[ií]ficos?               # específicos (con o sin tilde)
        (?:                                        # inicio del grupo opcional
            \s*\(
                (?:max\.\s*)?                      # "max." opcional
                \d+\s*p[aá]gina[s]?                # "1 página", "1página", plural
            \)
        )?                                         # <-- TODO el bloque entre paréntesis es opcional
        """,
        

        # "debe ser suficientemente informativo ... proyecto:"
        r"""
        debe\s+ser\s+suficientemente\s+informativo
        [\s\S]{0,800}?             # tolera contenido intermedio, incl. saltos de línea
        proyecto:
        """,

        # “problema que se abordará, objetivos, metodología y resultados que se esperan ... (de evaluadores | de la investigación)”
        r"""
        problema\s+que\s+se\s+abordar[áa]
        (?:[\s\S]{0,400}?objetivos)?            # <- OPCIONAL
        [\s\S]{0,400}?metodolog[ií]a
        [\s\S]{0,400}?resultados\s+que\s+se\s+esperan
        [\s\S]{0,400}?
        (?:de\s+evaluadores|de\s+la?\s+investigaci[oó]n)
        (?:\s*,?\s*etc\.)?                      # <- opcional “etc.”
        """,
        
        # “debe considerarse que un resumen bien formulado facilita ... evaluadores”
        r"""
        debe\s+considerarse\s+que\s+un\s+resumen\s+bien\s+formulado\s+facilita
        [\s\S]{0,400}?evaluadores
        """,

        # Bloques en inglés (tal como los tenías, pero robustecidos)
        r"""
        DESCRIBE\s+THE\s+MAIN\s+ISSUES\s+TO\s+BE\s+ADDRESSED
        [\s\S]{0,1000}?EXPECTED\s+RESULTS\.
        """,
        r"""
        THE\s+MAXIMUM\s+LENGTH\s+FOR\s+THIS\s+SECTION
        [\s\S]{0,1000}?SIMILAR\)\.
        """,
        r"""
        AVOID\s+INCLUDING\s+IN\s+THIS\s+SECTION\s+INFORMATION
        [\s\S]{0,1000}?BACKGROUNDS\.
        """,
        ####### Textos únicos identificados (no se repiten en varias casillas): 
        # Caso 1: versión extendida enviada a fondecyt 2022
        r"""(?ixs)
        \(\s*una\s+versi[oó]n\s+extendida.*?
        fondecyt\s+de\s+iniciaci[oó]n\s+2022.*?
        idioma\s+ingl[eé]s.\s*\)
        """,

        # Caso 2: señalar el proyecto, objetivos y diferencia sg1/sg2
        r"""(?ixs)
        se[nñ]alar\s+el\s+del\s+proyecto.*?
        objetivos.*?
        fondecyt.*?
        sg1.*?sg2
        """
    ]

    flags = re.IGNORECASE | re.DOTALL | re.VERBOSE
    return [re.compile(p, flags=flags) for p in pats]

def _short_pat(pat: str, n: int = 50) -> str:
    pat = pat.strip()
    return (pat[:n] + "...") if len(pat) > n else pat

def check_deleted_expressions(texts, return_long: bool = False):
    """
    Detecta (sin eliminar) qué fragmentos coincidirían con los patrones de
    get_expressions_to_delete() en cada texto.

    Parámetros
    ----------
    texts : iterable[str]
        Lista/iterable de textos.
    return_long : bool, opcional (default=False)
        Si True, además del pivote ancho devuelve un DataFrame largo con una fila por match.

    Devuelve
    --------
    df_wide : pd.DataFrame
        Filas = text_idx, Columnas = patrón (string acortado), Valores = coincidencias concatenadas.
    df_long (opcional) : pd.DataFrame
        Columnas: text_idx, pattern_id, pattern, start, end, match.
    """
    # Preparar patrones
    patterns = get_expressions_to_delete()
    pat_meta = [
        {"id": f"P{i+1}", "obj": p, "name": _short_pat(p.pattern)}
        for i, p in enumerate(patterns)
    ]

    # Recolectar matches
    records = []
    for i, raw_text in enumerate(texts):
        # Normalización básica (sin eliminar nada por regex)
        text = raw_text if isinstance(raw_text, str) else ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200F\u202F\u205F\u3000\uFEFF]', ' ', text)
        text = re.sub(r'\_x000D_', ' ', text)

        for meta in pat_meta:
            pat = meta["obj"]
            for m in pat.finditer(text):
                match_txt = m.group(0)
                records.append({
                    "text_idx": i,
                    "pattern_id": meta["id"],
                    "pattern": meta["name"],
                    "start": m.start(),
                    "end": m.end(),
                    "match": match_txt.strip()
                })

    # Si no hubo coincidencias, devolver DataFrame vacío consistente
    if not records:
        df_wide = pd.DataFrame(columns=["text_idx"] + [m["name"] for m in pat_meta])
        df_wide.set_index("text_idx", inplace=True)
        return (df_wide, pd.DataFrame(columns=["text_idx", "pattern_id", "pattern", "start", "end", "match"])) if return_long else df_wide

    # DF largo
    df_long = pd.DataFrame(records)

    # Agregar múltiples matches por (text_idx, pattern) y concatenar sin duplicados preservando orden
    agg = (
        df_long.groupby(["text_idx", "pattern"], as_index=False)["match"]
        .apply(lambda s: " | ".join(dict.fromkeys([x for x in s if x])))
    )

    # Pivot a formato ancho
    df_wide = agg.pivot(index="text_idx", columns="pattern", values="match").fillna("")

    # Asegurar columnas para todos los patrones (aunque queden vacías)
    all_cols = [m["name"] for m in pat_meta]
    for c in all_cols:
        if c not in df_wide.columns:
            df_wide[c] = ""
    df_wide = df_wide.reindex(columns=all_cols).sort_index()

    if return_long:
        return df_wide, df_long.sort_values(["text_idx", "start"])
    return df_wide

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalizar caracteres unicode (acentos, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Estandarización de carácteres de espacio
    text = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200F\u202F\u205F\u3000\uFEFF]', ' ', text)
    text = re.sub(r'\_x000D_', ' ', text)

    # Eliminar instrucciones comunes del formulario
    for pattern in get_expressions_to_delete():
        text = pattern.sub(" ", text)

    # Eliminar referencias tipo [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)

    # Eliminar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Eliminar números tipo "1. Introducción"
    text = re.sub(r'\d+\.\s*[A-Za-zÁÉÍÓÚáéíóúñÑ]+', '', text)

    # Eliminar caracteres no alfanuméricos (excepto puntuación básica)
    #text = re.sub(r'[^\w\s.,;:()\[\]¿?!¡%\-\\n]', '', text)

    # Eliminar puntuación que se encuentre al principio de un párrafo
    text = re.sub(r'^[\s:.,()]+', '', text)

    # Eliminar múltiples espacios
    text = re.sub(r'[ \t]+', ' ', text)

    #busca cualquier secuencia de 3 o más saltos de línea consecutivos.
    text = re.sub(r'\n{3,}', '\n', text)
    
    #Expandir acronimos
    #text = expand_acronyms(text)

    return text.strip().lower()

#Esta función aún no funciona completamente bien, así que no está en el Pipeline
def expand_acronyms(text):
    acronyms = {}
    # Encontrar definiciones de acrónimos tipo "Texto largo (ACR)"
    pattern = re.compile(r'\b([A-Z][A-Za-z0-9&.\s]+?)\s*\(\s*([A-Z]{2,})\s*\)')
    for match in pattern.finditer(text):
        long_form, short_form = match.groups()
        acronyms[short_form] = long_form.strip()

    # Eliminar la definición original dejando solo la forma larga
    text = pattern.sub(lambda m: m.group(1), text)

    # Reemplazar todas las apariciones del acrónimo por la forma larga
    if acronyms:
        acronym_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, acronyms.keys())) + r')\b')
        text = acronym_pattern.sub(lambda m: acronyms[m.group(0)], text)

    return text



def one_hot_codification(df, col_name, class_names):
    """
    Codifica en one-hot una columna categórica y agrega columnas binarias (0/1).

    Inputs:
    - df (pd.DataFrame): DataFrame con los datos.
    - col_name (str): Nombre de la columna categórica a codificar.
    - class_names (list): Lista de clases a transformar en columnas.

    Output:
    - df (pd.DataFrame): DataFrame con columnas nuevas (una por clase en class_names).
    """
    for name in class_names: 
        labels = np.isin(df[col_name], name)
        labels = np.where(labels, 1, 0).astype(int)
        df[str(name)] = labels
    return df
