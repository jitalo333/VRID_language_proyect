import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

#Dejar estas funciones en utils de visualization
def print_cm(y, preds):
    #Confusion matrix
    cm = confusion_matrix(y, preds, normalize="true")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title("Matriz de Confusión")
    plt.colorbar()

    # Agregar valores dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.3f}",
                    ha="center", va="center", color="black")

    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

def eval_model_print_cm(best_model, X_test, y_test, lang_es):
  results = {}
  preds = best_model.predict(X_test)
  cm = confusion_matrix(y_test, preds)
  results = {
      'accuracy': accuracy_score(y_test, preds),
      'f1_macro': f1_score(y_test, preds, zero_division=0, average="macro"),
      'cm': cm,
  }
  print_cm(y_test, preds)
  return results