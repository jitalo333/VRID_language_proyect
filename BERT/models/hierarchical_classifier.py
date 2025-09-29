from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from models.specter import embed_texts
from models.similarity import build_ocde_texts
import numpy as np

def sim_vectors(pub_vec, ocde_vecs):
    sims = ocde_vecs @ pub_vec
    return sims

def read_ocde_areas_from_dic(ocde_keys):
    ocde_area = []
    for key, _ in ocde_keys:
        ocde_area.append(key)

    # únicos en orden de aparición
    ocde_area = list(dict.fromkeys(ocde_area))

    return ocde_area

class subarea_hierarchical_classifier: 
    def __init__(self, models_dict, BASE_MODEL, ADAPTER_NAME, ocde_hierarchy):
        self.models_dict = models_dict
        self.BASE_MODEL = BASE_MODEL
        self.ADAPTER_NAME = ADAPTER_NAME
        #Generate ocde texts
        self.ocde_keys, ocde_texts = build_ocde_texts(ocde_hierarchy)
        #Generate embeddings 
        self.emb_ocde = embed_texts(ocde_texts, self.BASE_MODEL, self.ADAPTER_NAME)
        #Generar areas OCDE
        self.ocde_area = read_ocde_areas_from_dic(self.ocde_keys)
    
    def predict_proba_area(self, embedding):
        all_prob = []
        for key, model in self.models_dict.items():
            prob = model.predict_proba(embedding)  # shape (n_samples, 2)
            model_prob = np.where(prob[:, 0] > prob[:, 1], 0, prob[:, 1])
            all_prob.append(model_prob)

        return np.column_stack(all_prob)  # transponer en el mismo paso
    
    def compute_similarity(self, embedding, emb_ocde, use_cosine=True):
        if use_cosine is not None:
            return cosine_similarity(embedding, emb_ocde)  # (n_samples, n_ocde)
        else:
            return np.array([sim_vectors(embed, emb_ocde) for embed in embedding])
    
    def compute_num_subkeys(self, ocde_hierarchy, ocde_area):
        num_subkeys = []
        for area in ocde_area:
            n = len(ocde_hierarchy[area])
            #print(area, n)
            num_subkeys.append(n)
        return np.array(num_subkeys)
    
    def predict_subarea_proba(self, texts, use_softmax = None):
        embedding = embed_texts(texts, self.BASE_MODEL, self.ADAPTER_NAME)
        #Predexir area OCDE con modelo de ML 
        prob = self.predict_proba_area(embedding)
        #Cálculo de similaridad entre textos y subcategorías OCDE
        similarity_matrix = self.compute_similarity(embedding, self.emb_ocde)
        #Número de subclaves por area OCDE
        num_subkeys = self.compute_num_subkeys(self.ocde_hierarchy, self.ocde_area)
        #Se expande la matriz de predicciones de modelos de ML utilizando cantidad de claves por modelo
        prob_expanded = np.repeat(prob, num_subkeys, axis=1)
    
        #Se utiliza matriz expandida para eliminar de selección subareas OCDE que no hayan 
        #Sido seleccionadas por modelos de ML
        if use_softmax is not None:
            similarity_matrix = softmax(similarity_matrix)
            
        new_similarity = similarity_matrix*prob_expanded
        #Se obtienen posiciones de subareas OCDE seleccionadas
        best_idx = np.argmax(new_similarity, axis=1)
        rea, sub = [], []
      
        for idx in best_idx:
            rea1, sub1 = self.ocde_keys[idx]
            rea.append(rea1)
            sub.append(sub1)

        return np.array(rea), np.array(sub)  
   