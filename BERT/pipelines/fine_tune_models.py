#Pytorch
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import inspect
from torch.optim import AdamW
from transformers import get_scheduler
#Optuna
from torch.utils.data import DataLoader
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.dataset import CvCustom, TextDataset
#mlflow
import mlflow
import git
import os

#Pytorch pipeline
def get_sample_weights_loss(y):
  y = np.asarray(y, dtype=np.int64)
  class_counts = np.bincount(y)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum()

  return class_weights

def unfreeze_last_layers(model, n_unfreeze: int):
    """
    Descongela las últimas `n_unfreeze` capas de un modelo Hugging Face.
    Compatible con BERT, RoBERTa, DistilBERT, ALBERT, XLM-R, etc.

    Args:
        model (torch.nn.Module): Modelo Hugging Face (posiblemente envuelto en DataParallel).
        n_unfreeze (int): Número de capas a descongelar.

    Returns:
        None. Modifica el modelo en su lugar.
    """

    # Si el modelo está envuelto en DataParallel, acceder al .module
    model_to_unfreeze = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Detectar backbone automáticamente
    backbone = None
    for attr in ["bert", "roberta", "distilbert", "albert", "xlm_roberta"]:
        if hasattr(model_to_unfreeze, attr):
            backbone = getattr(model_to_unfreeze, attr)
            break

    if backbone is None:
        raise AttributeError("❌ No se encontró un backbone conocido (bert/roberta/distilbert/albert/xlm_roberta).")

    print("backbone", backbone)
    
    # Obtener capas del encoder
    if hasattr(backbone.encoder, "layer"):
        encoder_layers = backbone.encoder.layer
    elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
        encoder_layers = backbone.transformer.layer  # DistilBERT
    else:
        raise AttributeError("❌ No se encontró el atributo 'layer' en el encoder del backbone.")

    # Descongelar últimas n capas
    for layer in encoder_layers[-n_unfreeze:]:
        for p in layer.parameters():
            p.requires_grad = True

class Pytorch_Pipeline():
    def __init__(self, model_class, sample_weights_loss=None, max_epochs = 200, use_scheduler=None):
        #Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Modelo
        self.model_class = model_class
        self.model = None
        #Elementos del entrenamiento
        self.params = None
        self.sample_weights_loss = sample_weights_loss
        self.criterion = None
        self.optimizer = None
        self.batch_size = None
        self.scheduler=None
        self.max_epochs = max_epochs
        #scheduler
        self.use_scheduler=use_scheduler
        #Best model
        self.best_model_state=None

    def partial_fit(self, loader):
        self.model.to(self.device)
        self.model.train()
        
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            out = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = out.logits
            loss = self.criterion(logits, batch["labels"].to(self.device))
            loss.backward()
            self.optimizer.step()
            if self.use_scheduler is not None:
                self.scheduler.step()

        return self

    def predict(self, loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                # mover batch al device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                xb = {k: v for k, v in batch.items() if k != "labels"}
                yb = batch["labels"]

                outputs = self.model(**xb)
                logits = outputs.logits

                # predicciones
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        y_pred = torch.cat(all_preds).numpy()
        return y_pred

    def predict_and_evaluate(self, loader):
            self.model.eval()
            total_loss = 0.0
            total_samples = 0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for batch in loader:
                    # mover batch al device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    xb = {k: v for k, v in batch.items() if k != "labels"}
                    yb = batch["labels"]

                    outputs = self.model(**xb)
                    logits = outputs.logits

                    # calcular pérdida (soporta reduction='mean' o 'none')
                    loss_val = self.criterion(logits, yb)
                    if loss_val.dim() > 0:              # p.ej., reduction='none' -> [B]
                        batch_loss = loss_val.mean()
                    else:
                        batch_loss = loss_val

                    bs = yb.size(0)
                    total_loss += batch_loss.item() * bs  # acumular ponderado por tamaño de batch
                    total_samples += bs

                    # predicciones
                    preds = logits.argmax(dim=1)

                    all_preds.append(preds.cpu())
                    all_targets.append(yb.cpu())

            avg_val_loss = total_loss / max(total_samples, 1)
            y_true = torch.cat(all_targets).numpy()
            y_pred = torch.cat(all_preds).numpy()
            f1 = f1_score(y_true, y_pred, average='weighted')

            return avg_val_loss, f1, y_true, y_pred

    def set_params(self, multi_GPU_on=None, **params):
        self.params = params

        # Obtener los parámetros esperados por el constructor de model_class
        #signature = inspect.signature(self.model_class.__init__)
        #valid_keys = set(signature.parameters.keys()) - {'self'}

        # Filtrar los params para incluir solo los esperados
        #filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        #self.model = self.model_class(**filtered_params)
        
        self.model = self.model_class
        if torch.cuda.device_count() > 1 and multi_GPU_on is not None:
            print("Usando", torch.cuda.device_count(), "GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params['lr']) 
        self.batch_size = self.params['batch_size']

        # Si el modelo está envuelto en DataParallel, accedemos al .module
        unfreeze_last_layers(self.model, self.params["n_unfreeze"])

    def get_params(self):
        return self.params
              
    def set_criterion(self, y):
          # ----------- Criterion -----------
          if self.sample_weights_loss is not None:
              class_weights = get_sample_weights_loss(y)
              class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
              self.criterion = nn.CrossEntropyLoss(weight=class_weights)
          else:
              self.criterion = nn.CrossEntropyLoss()

          return self

    def fit_early_stopping(self, train_loader, val_loader, labels):
        #Establecer criterion con sample weights si se especifica
        self.set_criterion(labels)
        self.best_model_state = None
        # ---------- Early stopping (por pérdida) ----------
        patience = 10
        min_delta = 1e-4
        best_val_loss = float('inf')
        epochs_no_improve = 0
        #scheduler
        num_training_steps = len(train_loader) * self.max_epochs
        if self.use_scheduler is not None:
            self.scheduler = get_scheduler(
                "linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )
        #Entrenamiento
        for epoch in range(self.max_epochs):
            self.partial_fit(train_loader)
            avg_val_loss, f1, _, _ = self.predict_and_evaluate(val_loader)
            
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
            print("f1:", f1)
        return f1
    
    def eval_test(self, model_dict, loader):
        all_preds, all_targets = [], []
        model = self.model_class
        model.load_state_dict(model_dict)
        with torch.no_grad():
            for batch in loader:
                # mover batch al device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                xb = {k: v for k, v in batch.items() if k != "labels"}
                yb = batch["labels"]

                outputs = model(**xb)
                logits = outputs.logits

                # predicciones
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'cm': confusion_matrix(y_true, y_pred)
        }
        return metrics
    
    def update_to_best_model(self):
        model = self.model_class
        model.load_state_dict(self.best_model_state)
        self.model = model

#Optuna model
def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(x) for x in obj]
    elif isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj

def get_metrics(y_true, y_pred, verbose = True):
  metrics = {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'f1_score': f1_score(y_true, y_pred, average='weighted'),
      'cm': confusion_matrix(y_true, y_pred)
  }
  if verbose:
    print(metrics)
  return metrics

class optuna_objective_cv:
    def __init__(self, X, y, n_classes, model_name, df_decode, SMOTE_on=None, sample_weights_loss=None, Test_mode = None):
        self.results = {}
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.sample_weights_loss = sample_weights_loss
        self.max_epochs = 200
        self.best_model_trial = None
        self.Test_mode = Test_mode
        self.df_decode = df_decode
        #BERT models
        self.model_name = model_name
    
    def get_loaders(self, X_train, X_test, y_train, y_test, batch_size):
        train_dataset = TextDataset(list(X_train), y_train, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        test_dataset = TextDataset(list(X_test), y_test, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        return train_loader, test_loader

    def objective(self, trial):
        # ----------- Hiperparámetros a optimizar -----------
        params={
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "batch_size":12,
        "n_unfreeze":trial.suggest_int("n_unfreeze", 1, 12)
        }
    
        #------------- StratifiedKFold -------------------------------
        F1 = []
        all_metrics = []
        cv_function=CvCustom(self.df_decode)
        for fold, (train_index, test_index) in enumerate(cv_function.split(self.X)):
            #---------------Split data-------------------------------
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            #--------------def model----------------------------------
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            pipeline_mlp =  Pytorch_Pipeline(model_class=model, sample_weights_loss = self.sample_weights_loss)
            #Set params
            pipeline_mlp.set_params(**params)
            #Set criterion
            pipeline_mlp.set_criterion(y_train)

            # ------------- Loaders --------------------
            train_loader, test_loader = self.get_loaders(X_train, X_test, y_train, y_test, pipeline_mlp.batch_size)
            # ---------- Early stopping (por loss) ----------
            patience = 10
            min_delta = 1e-4
            best_val_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(pipeline_mlp.max_epochs):
                pipeline_mlp.partial_fit(train_loader)
                avg_val_loss, f1, y_test, y_pred = pipeline_mlp.predict_and_evaluate(test_loader)
                # ---------- Optuna pruning con F1 ----------
                #Prune only on the first fold
                if fold == 0:
                    trial.report(f1, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                # ---------- Early stopping (por loss) ----------
                if avg_val_loss + min_delta < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = pipeline_mlp.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            #---------------- Save final result ----------------
            F1.append(f1)
            #-------------Visualization metrics-----------------
            metrics = get_metrics(y_test, y_pred)
            all_metrics.append(metrics)

        #------------ Compute avg among 5 folds ----------------
        mean_F1 = np.mean(F1)

        # ---------- Guarda el modelo del mejor trial según F1 ----------
        try:
            if trial.number == 0 or mean_F1 > trial.study.best_value:
                self.results = {
                    'metrics': self.avg_metrics(all_metrics),
                    'best_params': params,
                    'model_state_dict': best_model_state,
                    'epoch_number': epoch
                }

        except ValueError:
          pass

        return mean_F1
    
    def avg_metrics(self, all_metrics):
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            values = [metrics[metric] for metrics in all_metrics]

            if metric == 'cm':
                avg_metrics[metric] = np.mean(values, axis=0).astype(int)  # o float si prefieres
            else:
                avg_metrics[metric] = np.mean(values)
        return avg_metrics

    #----------- Método para obtener los resultados -----------
    def get_results(self):
      return self.results
    
#Mlflow functions
def metrics_lang(y, preds, lang_es):
    #Conversión en array
    y = np.array(y)
    preds = np.array(preds)
    lang_es = np.array(lang_es)

    # Seleccionar por máscara booleana
    y_es = y[lang_es] #Data originalmente en español
    preds_es = preds[lang_es]

    y_en = y[~lang_es] #Data originalmente en inglés
    preds_en = preds[~lang_es]
    
    #Computo de métricas
    f1_es = f1_score(y_es, preds_es, average="weighted")
    f1_en = f1_score(y_en, preds_en, average="weighted")
    cm_es = confusion_matrix(y_es, preds_es)
    cm_en = confusion_matrix(y_en, preds_en)

    return f1_es, f1_en, cm_es, cm_en

def eval_model(pipeline_pytorch, test_loader, y_test, lang_es):
  results = {}
  preds = pipeline_pytorch.predict(test_loader)
  cm = confusion_matrix(y_test, preds)
  f1_es, f1_en, cm_es, cm_en = metrics_lang(y_test, preds, lang_es)
  #t_n, f_p, f_n, t_p = cm()
  results = {
      'accuracy': accuracy_score(y_test, preds),
      'precision': precision_score(y_test, preds, zero_division=0),
      'recall': recall_score(y_test, preds, zero_division=0),
      'f1_macro': f1_score(y_test, preds, zero_division=0, average="macro"),
      'cm': cm,
      'f1_es': f1_es,
      'f1_en': f1_en,
      'cm_es': cm_es,
      'cm_en': cm_en
  }
  return results, preds

def safe_log_metric(name, value):
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            if np.size(value) == 1:
                value = float(np.array(value).item())
            else:
                raise ValueError("Métrica con más de un valor.")
        else:
            value = float(value)
        mlflow.log_metric(name, value)
    except Exception as e:
        print(f"⚠️ No se pudo loggear {name}: {e}")

def mlflow_ckeckpoint(exp_info, pipeline_pytorch, extra_parms, test_loader, y_test, df_test, mode="server"):
    
    if mode == "server":
        # Set backend store
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri)) 
    
    elif mode == "local": 
        # Set backend store
        mlflow.set_tracking_uri(exp_info["tracking_path"])
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri)) 

        # Verificar si existe experimento, si no crearlo
        experiment = mlflow.get_experiment_by_name(exp_info["exp_name"])

        if experiment is None:
            exp_id = mlflow.create_experiment(
                exp_info["exp_name"],
                artifact_location=exp_info["artifact_path"]
            )
            print(f"Experimento creado con ID: {exp_id}")
        else:
            exp_id = experiment.experiment_id
            print(f"Experimento ya existe con ID: {exp_id}")
    
    else: 
        print("Especificar modo de almacenamiento")
        return 0

    # Define el experimento (lo crea si no existe)
    mlflow.set_experiment(exp_info["exp_name"])
    
    # Obtener commit actual
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha

    with mlflow.start_run(run_name=exp_info["run_name"]):
        print(f"📝 Registrando modelo en MLflow: {exp_info['run_name']}")

        # Hiperparámetros
        try:
            mlflow.log_params(pipeline_pytorch.get_params())
        except:
            print(f"⚠️ No se pudieron loggear los hiperparámetros para {exp_info['run_name']}")

        #Parámetros adicionales
        for k, v in extra_parms.items():
            mlflow.log_param(k, v)

        # Métricas de test
        results_test, preds = eval_model(pipeline_pytorch, test_loader, y_test, df_test["Español"])
        for k, v in results_test.items():
            if k.startswith("cm"):
                # Guardar confusion matrix (o similar) como artefacto
                # Guardar como CSV temporal
                fname = f"{k}.csv"
                np.savetxt(fname, v, delimiter=",", fmt="%d")

                mlflow.log_artifact(fname, artifact_path="confusion_matrices")

                # Eliminar archivo local si no lo necesitas
                os.remove(fname)

            else:
                # Guardar métrica numérica
                safe_log_metric(f"test_{k}", v)
        
        #Guardar dataframe con predicciones
        fname = f"df_test_preds.csv"
        df_test["preds"]=preds
        df_test["y_test"]=y_test
        df_test.to_csv(fname, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(fname, artifact_path="predictions")
        os.remove(fname)
        
        #Guardar commit de git
        mlflow.log_param("git_commit", commit_hash)

        #Guardar plot de optuna
                
        # Guardar modelo
        mlflow.pytorch.log_model(pipeline_pytorch.model, name = "model")
  