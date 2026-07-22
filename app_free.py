"""
TimeCapsule - FREE Version (Streamlit)
------------------------------------------
Versi ini didesain 100% GRATIS untuk keperluan belajar/tugas kuliah.
Tidak ada API berbayar sama sekali.

Teknologi yang dipakai (semua gratis):
- Google Gemini API (gemini-2.5-flash)       -> generate prompt teks (free tier)
- Google Gemini API (gemini-2.5-flash-image) -> generate gambar AI dari foto (free tier, "Nano Banana")
- OpenCV (lokal, tanpa API)                   -> "video" dari efek Ken Burns (zoom/pan) di atas gambar AI

CATATAN PENTING:
Tidak ada API video-generation (motion AI beneran) yang gratis di pasaran
(Runway, fal.ai/Seedance, Veo, Kling semuanya berbayar karena butuh compute GPU besar).
Sebagai gantinya, "video" di versi ini dibuat dengan animasi zoom/pan sederhana
di atas gambar AI yang sudah dihasilkan Gemini -> teknik umum "photo to video"
yang sepenuhnya lokal & gratis, cocok untuk demo/tugas kuliah.

Cara dapat GOOGLE_API_KEY (gratis, tanpa kartu kredit):
1. Buka https://aistudio.google.com/apikey
2. Login dengan akun Google
3. Klik "Create API key", copy hasilnya

Jalankan:
    streamlit run app_free.py

Install dependencies:
    pip install streamlit pillow numpy opencv-python-headless google-genai python-dotenv
"""

import os
import io
import uuid
from datetime import datetime
from enum import Enum

import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

os.makedirs("intermediate_images", exist_ok=True)
os.makedirs("final_videos", exist_ok=True)


# ----------------------------------------------------------------------------
# Pilihan-pilihan
# ----------------------------------------------------------------------------
class EthnicityOptions(Enum):
    CAUCASIAN = "Caucasian"
    AFRICAN = "African"
    ASIAN = "Asian"
    HISPANIC = "Hispanic"
    MIDDLE_EASTERN = "Middle Eastern"
    MIXED = "Mixed Heritage"


class TimePeriodOptions(Enum):
    ANCIENT = "Ancient Times (Before 500 AD)"
    MEDIEVAL = "Medieval (500-1500 AD)"
    RENAISSANCE = "Renaissance (1400-1600)"
    COLONIAL = "Colonial Era (1600-1800)"
    VICTORIAN = "Victorian Era (1800-1900)"
    MODERN = "Modern Era (1990-Present)"
    FUTURISTIC = "Futuristic (Near Future)"


class ProfessionOptions(Enum):
    WARRIOR = "Warrior/Soldier"
    SCHOLAR = "Scholar/Teacher"
    ARTISAN = "Artisan/Craftsperson"
    EXPLORER = "Explorer/Adventurer"
    NOBLE = "Noble/Aristocrat"


class AnimationOptions(Enum):
    ZOOM_IN = "Zoom In (perlahan mendekat)"
    ZOOM_OUT = "Zoom Out (perlahan menjauh)"
    PAN_LEFT_RIGHT = "Pan Kiri ke Kanan"


# ----------------------------------------------------------------------------
# Fungsi inti - Gemini (gratis)
# ----------------------------------------------------------------------------
def make_image_prompt(ethnicity, time_period, profession) -> str:
    """Buat prompt gambar dengan Gemini (gratis), fallback ke prompt sederhana kalau gagal."""
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        base_prompt = (
            f"Create a simple, clean prompt for AI image generation: "
            f"Ethnicity: {ethnicity.value}, Profession: {profession.value}, "
            f"Time period: {time_period.value}. "
            f"Show appropriate clothing and setting, unique background, "
            f"appropriate for all ages. Maximum 30 words. Only output the prompt itself."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=base_prompt,
        )
        return response.text.strip()
    except Exception as e:
        st.warning(f"Gemini gagal membuat prompt, pakai fallback. Detail: {e}")
        return f"{ethnicity.value} {profession.value} from {time_period.value}, professional portrait"


def generate_image_with_gemini(photo_path, prompt):
    """Generate gambar baru dari foto referensi + prompt menggunakan Gemini 2.5 Flash Image (Nano Banana, gratis)."""
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    ref_image = Image.open(photo_path)

    full_prompt = (
        f"Using the person in the reference photo, generate a new portrait image: {prompt}. "
        f"Keep the person's face and identity consistent with the reference photo."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[full_prompt, ref_image],
    )

    image_bytes = None
    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None):
            image_bytes = part.inline_data.data
            break

    if not image_bytes:
        raise RuntimeError("Gemini tidak mengembalikan gambar. Coba ubah prompt atau foto.")

    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.png"
    filepath = os.path.join("intermediate_images", filename)
    Image.open(io.BytesIO(image_bytes)).save(filepath)
    return filepath


def create_video_from_image(image_path, animation: "AnimationOptions", duration_sec=5, fps=24):
    """
    Buat 'video' dari satu gambar dengan efek Ken Burns (zoom/pan), 100% lokal, tanpa API.
    """
    import cv2

    img = Image.open(image_path).convert("RGB")
    out_w, out_h = 1280, 720

    # Perbesar gambar dulu supaya ada ruang gerak untuk efek zoom/pan
    scale_factor = 1.3
    big_w, big_h = int(out_w * scale_factor), int(out_h * scale_factor)
    img_resized = img.resize((big_w, big_h))
    frame_big = np.array(img_resized)  # RGB

    total_frames = int(duration_sec * fps)
    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.mp4"
    filepath = os.path.join("final_videos", filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (out_w, out_h))

    max_x_offset = big_w - out_w
    max_y_offset = big_h - out_h

    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)  # 0.0 -> 1.0

        if animation == AnimationOptions.ZOOM_IN:
            # geser dari pojok kiri-atas (0) ke tengah, makin 'zoom in' terasa
            x_off = int((max_x_offset / 2) * t)
            y_off = int((max_y_offset / 2) * t)
        elif animation == AnimationOptions.ZOOM_OUT:
            x_off = int((max_x_offset / 2) * (1 - t))
            y_off = int((max_y_offset / 2) * (1 - t))
        else:  # PAN_LEFT_RIGHT
            x_off = int(max_x_offset * t)
            y_off = max_y_offset // 2

        crop = frame_big[y_off:y_off + out_h, x_off:x_off + out_w]
        frame_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return filepath


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TimeCapsule (Gratis)", page_icon="🕰️")
st.title("🕰️ TimeCapsule - Versi Gratis")
st.caption("100% gratis: pakai Google Gemini free tier untuk gambar, animasi lokal untuk video.")

with st.sidebar:
    st.subheader("Status API Key")
    st.write(("✅ " if GOOGLE_API_KEY else "❌ ") + "GOOGLE_API_KEY")
    st.caption("Dapatkan gratis di https://aistudio.google.com/apikey")

uploaded_photo = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    ethnicity_label = st.selectbox("Etnis", [e.value for e in EthnicityOptions])
    profession_label = st.selectbox("Profesi", [p.value for p in ProfessionOptions])
with col2:
    time_period_label = st.selectbox("Periode Waktu", [t.value for t in TimePeriodOptions])
    animation_label = st.selectbox("Efek Video", [a.value for a in AnimationOptions])

ethnicity = next(e for e in EthnicityOptions if e.value == ethnicity_label)
time_period = next(t for t in TimePeriodOptions if t.value == time_period_label)
profession = next(p for p in ProfessionOptions if p.value == profession_label)
animation = next(a for a in AnimationOptions if a.value == animation_label)

if st.button("🚀 Generate (Gratis)", type="primary", use_container_width=True):
    if not uploaded_photo:
        st.error("Upload foto dulu ya.")
        st.stop()
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY belum diisi. Ambil gratis di https://aistudio.google.com/apikey")
        st.stop()

    photo_path = os.path.join("intermediate_images", f"input_{uuid.uuid4()}.jpg")
    Image.open(uploaded_photo).convert("RGB").save(photo_path)

    with st.status("Membuat TimeCapsule kamu...", expanded=True) as status:
        st.write("Membuat prompt gambar dengan Gemini...")
        image_prompt = make_image_prompt(ethnicity, time_period, profession)
        st.write(f"→ Prompt: _{image_prompt}_")

        st.write("Generate gambar dengan Gemini 2.5 Flash Image (Nano Banana)...")
        try:
            image_path = generate_image_with_gemini(photo_path, image_prompt)
        except Exception as e:
            status.update(label="Gagal generate gambar", state="error")
            st.error(f"Error: {e}")
            st.stop()

        st.write("Membuat animasi video dari gambar (lokal, tanpa API)...")
        try:
            video_path = create_video_from_image(image_path, animation)
        except Exception as e:
            status.update(label="Gagal membuat video", state="error")
            st.error(f"Error: {e}")
            st.stop()

        status.update(label="Selesai! 🎉", state="complete")

    st.subheader("Hasil")
    c1, c2 = st.columns(2)
    with c1:
        st.image(image_path, caption="Gambar AI (Gemini)")
    with c2:
        st.video(video_path)
