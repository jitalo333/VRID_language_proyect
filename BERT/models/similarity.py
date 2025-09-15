import numpy as np

#Funciones necesarias 
def build_ocde_texts(ocde_dict):
    keys, texts = [], []
    for area, subs in ocde_dict.items():
        for sub, discs in subs.items():
            keys.append((area, sub))
            area_ctx = f"Field of research: {area}. "
            sub_ctx  = f"Specific research domain: {sub}. "
            disc_ctx = f"Related disciplines and topics: {'; '.join(discs)}"
            texts.append(area_ctx + sub_ctx + disc_ctx)
    return keys, texts

def classify_two_stage(pub_vec, ocde_vecs, ocde_keys, ocde_hierarchy, pub_text=""):
    sims = ocde_vecs @ pub_vec
    best_idx = int(np.argmax(sims))
    sim1 = float(sims[best_idx])
    area1, sub1 = ocde_keys[best_idx]
    
    result = area1
    
    return result

def sim_vectors(pub_vec, ocde_vecs, ocde_keys, ocde_hierarchy, pub_text=""):
    sims = ocde_vecs @ pub_vec
    return sims
