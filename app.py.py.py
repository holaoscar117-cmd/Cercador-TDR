import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

# ==========================================
# 1. EL TEU CODI EXACTE (SENSE CANVIS)
# ==========================================

def carregar_pdf(fitxer_pdf):
    reader = PdfReader(fitxer_pdf)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append({"pagina": i + 1, "text": text})
    return documents

def crear_fragments(documents, paraules_per_fragment=200, encavalcament=50):
    fragments = []
    for doc in documents:
        text = doc["text"]
        paraules = text.split()

        if len(paraules) <= paraules_per_fragment:
            fragments.append({"pagina": doc["pagina"], "text": text})
            continue

        i = 0
        while i < len(paraules):
            bloc = paraules[i : i + paraules_per_fragment]
            fragment_text = " ".join(bloc)
            fragments.append({"pagina": doc["pagina"], "text": fragment_text})
            i += paraules_per_fragment - encavalcament

    return fragments

def cercar(pregunta, model, vectors_text, fragments, top_k=1):
    vector_pregunta = model.encode([pregunta])
    similituds = np.dot(vectors_text, vector_pregunta.T).squeeze()
    index_millor = np.argmax(similituds)
    return fragments[index_millor]

# ==========================================
# 2. CARGA DEL MODEL (Amb memòria de Streamlit)
# ==========================================

@st.cache_resource
def carregar_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = carregar_model()

# ==========================================
# 3. INTERFÍCIE D'USUARI (STREAMLIT)
# ==========================================

st.title("🔍 Cercador de PDF amb Embeddings (TdR)")
st.write("Cerca informació dins d'un document PDF utilitzant vectors i similitud matemàtica.")

fitxer_pujat = st.file_uploader("Carrega un document PDF", type=["pdf"])

if fitxer_pujat is not None:
    # Processar el PDF
    documents = carregar_pdf(fitxer_pujat)
    fragments = crear_fragments(documents)
    
    st.success(f"PDF carregat correctament! S'han creat {len(fragments)} fragments.")
    
    # Generar vectors
    textos = [doc["text"] for doc in fragments]
    vectors_text = model.encode(textos)
    
    # Cerca de text
    pregunta = st.text_input("Quina informació vols cercar?")
    
    if pregunta:
        resultat = cercar(pregunta, model, vectors_text, fragments)
        st.subheader(f"📍 Trobat a la pàgina {resultat['pagina']}:")
        st.info(f"\"{resultat['text']}\"")