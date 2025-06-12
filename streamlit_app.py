import streamlit as st
import requests
from PIL import Image
import io

# Endpoints de l’API
API_URL = "https://self-driving-prediction-283057982875.europe-west1.run.app"
PREDICT_ENDPOINT = f"{API_URL}/predict/"
IMAGE_LIST_ENDPOINT = f"{API_URL}/image_ids/"
REAL_IMAGE_ENDPOINT = f"{API_URL}/image/"
REAL_MASK_ENDPOINT = f"{API_URL}/mask/"

st.set_page_config(page_title="Segmentation U-Net", layout="wide")
st.title("🔍 Segmentation automatique d'image avec U-Net")

st.markdown("## 🖼️ Choisissez une image à segmenter parmi la base de données")

# Obtenir la liste des images disponibles
try:
    ids_response = requests.get(IMAGE_LIST_ENDPOINT, timeout=10)
    ids_response.raise_for_status()
    image_ids = ids_response.json()
except Exception as e:
    st.error(f"Erreur lors de la récupération des IDs : {e}")
    st.stop()

# Sélection de l'ID d'image
selected_id = st.selectbox("📂 ID de l'image :", image_ids)

# Lancer la prédiction pour l’image sélectionnée
if st.button("🚀 Segmenter cette image"):
    with st.spinner("Chargement et prédiction en cours..."):

        try:
            # Obtenir l’image originale
            img_response = requests.get(f"{REAL_IMAGE_ENDPOINT}{selected_id}", timeout=10)
            img_response.raise_for_status()
            real_image = Image.open(io.BytesIO(img_response.content)).convert("RGB")

            # Obtenir le masque réel
            mask_response = requests.get(f"{REAL_MASK_ENDPOINT}{selected_id}", timeout=10)
            mask_response.raise_for_status()
            real_mask = Image.open(io.BytesIO(mask_response.content)).convert("RGB")

            # Envoyer l’image à l’API pour prédiction
            files = {"file": (f"{selected_id}.png", img_response.content, "image/png")}
            pred_response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
            pred_response.raise_for_status()
            pred_mask = Image.open(io.BytesIO(pred_response.content)).convert("RGB")

        except Exception as e:
            st.error(f"Erreur lors de la requête à l’API : {e}")
            st.stop()

        # Affichage des résultats
        st.markdown("## 📊 Résultats de segmentation")
        
        size = (400, 400)
        
        real_image_resized = real_image.resize(size)
        real_mask_resized = real_mask.resize(size)
        pred_mask_resized = pred_mask.resize(size)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Image originale**")
            st.image(real_image_resized)
        
        with col2:
            st.markdown("**Masque réel**")
            st.image(real_mask_resized)
        
        with col3:
            st.markdown("**Masque prédit**")
            st.image(pred_mask_resized)

        # Téléchargement
        buf = io.BytesIO()
        pred_mask.save(buf, format="PNG")
        st.download_button(
            label="📥 Télécharger le masque prédit",
            data=buf.getvalue(),
            file_name=f"mask_pred_{selected_id}.png",
            mime="image/png"
        )
