import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.image_preprocessing import ImagePreprocessor, FringeDetector

def render():
    st.header("📊 Traditional Photoelasticity Analysis")

    # Material selection
    materials = {
        "Polycarbonate": 6.3e-12,
        "Epoxy": 3.4e-12,
        "Bakelite": 2.9e-12,
        "Araldite": 4.0e-12,
        "Acrylic (PMMA)": 2.6e-12
    }
    material = st.selectbox("Select Material", list(materials.keys()))
    f_sigma = materials[material]
    st.write(f"**Stress-optic coefficient for {material}:** {f_sigma:.2e} m²/N")

    uploaded_file = st.file_uploader("Upload Photoelastic Image", type=['jpg', 'png', 'tif', 'bmp'])
    t = st.number_input("Specimen thickness t (mm)", value=3.0, step=0.1)
    lambda_light = st.number_input("Light wavelength λ (nm)", value=546.0, step=1.0)

    if uploaded_file and st.button("Run Analysis"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(img)
        img_preprocessed = result['preprocessed']

        detector = FringeDetector()
        fringe_results = detector.detect_fringes(img_preprocessed)

        st.success("✅ Processing Complete!")

        stages = {
            "Original": result['original'],
            "Preprocessed": img_preprocessed,
            "Frangi": fringe_results['frangi'],
            "Binary": fringe_results['binary'],
            "Skeleton": fringe_results['skeleton']
        }

        tab_titles = list(stages.keys())
        tabs = st.tabs(tab_titles)

        for i, (title, img_stage) in enumerate(stages.items()):
            with tabs[i]:
                st.write(f"**{title}**")
                fig, ax = plt.subplots()
                cmap = 'jet' if title == "Frangi" else 'gray'
                ax.imshow(img_stage, cmap=cmap)
                ax.axis('off')
                st.pyplot(fig)

if __name__ == "__main__":
    render()