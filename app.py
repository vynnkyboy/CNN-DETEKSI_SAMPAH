import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_sampah.keras")

model = load_model()

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Deteksi Sampah CNN",
    page_icon="🗑️",
    layout="centered"
)

# =============================
# STYLE UI
# =============================
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-top: 20px;
}
.badge-organik {
    background-color: #2ecc71;
    color: white;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
.badge-anorganik {
    background-color: #e74c3c;
    color: white;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🗑️ Deteksi Sampah Organik & Anorganik")
st.write(
    "Aplikasi ini mengimplementasikan **Convolutional Neural Network (CNN)** "
    "untuk melakukan klasifikasi sampah secara otomatis."
)

# =============================
# UPLOAD GAMBAR
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload gambar sampah (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =============================
# PROSES & PREDIKSI
# =============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang di-upload",
        use_container_width=True
    )

    # =============================
    # PREPROCESSING (SESUAI TRAINING)
    # =============================
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =============================
    # PREDIKSI
    # =============================
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    class_names = ["Organik", "Anorganik"]
    hasil = class_names[class_index]

    # =============================
    # CARD HASIL
    # =============================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🔍 Hasil Klasifikasi")

    if hasil == "Organik":
        st.markdown(
            "<span class='badge-organik'>🟢 ORGANIK</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span class='badge-anorganik'>🔴 ANORGANIK</span>",
            unsafe_allow_html=True
        )

    st.write("")
    st.write("📊 Tingkat Keyakinan Model")
    st.progress(float(confidence))
    st.write(f"**{confidence*100:.2f}%**")

    st.write("📌 Detail Probabilitas:")
    st.write(f"- Organik : {prediction[0][0]*100:.2f}%")
    st.write(f"- Anorganik : {prediction[0][1]*100:.2f}%")

    # =============================
    # SIMPAN GAMBAR KE MEMORY (PDF)
    # =============================
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # =============================
    # GENERATE PDF + GAMBAR
    # =============================
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    waktu = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Judul
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width / 2, height - 50,
                           "HASIL PREDIKSI CNN")

    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 80,
                           "Deteksi Sampah Organik dan Anorganik")

    # Informasi
    y = height - 130
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Tanggal           : {waktu}")
    y -= 20
    pdf.drawString(50, y, f"Jenis Sampah      : {hasil}")
    y -= 20
    pdf.drawString(50, y, f"Tingkat Keyakinan : {confidence*100:.2f} %")

    # Gambar di tengah
    img_reader = ImageReader(img_buffer)
    img_width = 260
    img_height = 260
    img_x = (width - img_width) / 2
    img_y = y - img_height - 20

    pdf.drawImage(
        img_reader,
        img_x,
        img_y,
        width=img_width,
        height=img_height,
        preserveAspectRatio=True,
        mask='auto'
    )

    # Probabilitas
    y = img_y - 40
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Probabilitas:")
    y -= 20

    pdf.setFont("Helvetica", 11)
    pdf.drawString(70, y, f"Organik    : {prediction[0][0]*100:.2f} %")
    y -= 20
    pdf.drawString(70, y, f"Anorganik  : {prediction[0][1]*100:.2f} %")

    # Footer
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawCentredString(
        width / 2,
        40,
        "Dokumen ini dihasilkan secara otomatis oleh sistem CNN"
    )

    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download Hasil Prediksi (PDF)",
        data=pdf_buffer,
        file_name="hasil_prediksi_sampah.pdf",
        mime="application/pdf"
    )

    st.markdown("</div>", unsafe_allow_html=True)
