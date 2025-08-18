import os
import shutil

def limpiar_drive_oculto(directorio):
    """
    Elimina de forma recursiva todas las carpetas ocultas típicas de Google Drive 
    que pueden causar errores con MLflow, como .trash, .config, .ipynb_checkpoints, etc.
    
    Parámetros:
    -----------
    directorio : str
        Ruta base donde buscar y eliminar carpetas ocultas
    """
    carpetas_a_borrar = [".ipynb_checkpoints", ".trash", ".config", ".TemporaryItems", ".Spotlight-V100", ".DS_Store"]

    for root, dirs, files in os.walk(directorio):
        for d in dirs:
            if d.startswith(".") or d in carpetas_a_borrar:
                path_completa = os.path.join(root, d)
                try:
                    shutil.rmtree(path_completa)
                    print(f"🧹 Eliminada carpeta: {path_completa}")
                except Exception as e:
                    print(f"⚠️ No se pudo eliminar {path_completa}: {e}")


limpiar_drive_oculto("/mlruns")
