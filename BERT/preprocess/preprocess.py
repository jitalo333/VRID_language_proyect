import pandas as pd
import re
import unicodedata


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
        una\s+versi[oó]n\s+extendida.*?
        fondecyt\s+de\s+iniciaci[oó]n\s+2022.*?
        idioma\s+ingl[eé]s
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

    # Eliminar puntiación que se encuentre al principio de un párrafo
    text = re.sub(r'^[\s:.,]+', '', text)

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
