import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import time
from gtts import gTTS
import io
import json
from google import genai
from google.genai import types

# ==========================================
# DATABASE FUNCTIONS
# ==========================================
def init_db():
    conn = sqlite3.connect('medikiosk.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, 
                  complaint TEXT, history TEXT, dosha TEXT, date TEXT, is_urgent BOOLEAN)''')
    conn.commit()
    conn.close()

def save_patient_record(name, age, complaint, history, dosha, is_urgent=False):
    conn = sqlite3.connect('medikiosk.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute('''INSERT INTO patients (name, age, complaint, history, dosha, date, is_urgent) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (name, age, complaint, history, dosha, date_str, is_urgent))
    conn.commit()
    conn.close()

def delete_patient(patient_id):
    conn = sqlite3.connect('medikiosk.db')
    c = conn.cursor()
    c.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# TRANSLATION ENGINE (GEMINI)
# ==========================================
def translate_with_gemini(text, target_language):
    """Sends text to Gemini for regional translation with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            prompt = f"Translate the following text exactly into {target_language}. Return ONLY the translated text without any markdown, quotes, or conversational filler:\n\n{text}"
            
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                time.sleep(3)
            else:
                return f"Gemini Translation Error: {e}"

# ==========================================
# USER INTERFACE SETUP
# ==========================================
st.set_page_config(page_title="MediKiosk", page_icon="🏥", layout="wide")
st.title("🏥 MediKiosk: AI Patient Intake")
st.markdown("Welcome to the self-service clinical intake platform.")

tab1, tab2, tab3, tab4 = st.tabs(["1. Registration", "2. Voice Intake", "3. Upload Reports", "4. Doctor Dashboard"])

# --- TAB 1: REGISTRATION ---
with tab1:
    st.header("Patient Details")
    with st.form("registration_form"):
        col1, col2 = st.columns(2)
        with col1:
            pat_name = st.text_input("Full Name")
        with col2:
            pat_age = st.number_input("Age", min_value=1, max_value=120, step=1)
        
        submit_reg = st.form_submit_button("Register Patient")
        if submit_reg and pat_name:
            st.session_state['current_patient'] = pat_name
            st.session_state['current_age'] = pat_age
            st.success(f"Registered {pat_name}. Please move to Voice Intake.")

# --- TAB 2: VOICE INTAKE (Gemini Assistant + Gemini Translation) ---
with tab2:
    st.header("🎤 Tell us your symptoms")
    
    patient_lang = st.selectbox(
        "Select your language / अपनी भाषा चुनें:", 
        ["Hindi", "English", "Bengali", "Tamil", "Telugu", "Marathi"]
    )
    
    st.markdown("Speak naturally into the microphone:")
    audio_value = st.audio_input("Record your symptoms")
    raw_symptoms = st.text_area("Or type them manually:", placeholder="e.g., Mujhe teen din se tez bukhar hai...")
    
    if st.button("Process with AI"):
        if 'current_patient' not in st.session_state:
            st.warning("Please register the patient in Tab 1 first.")
        elif audio_value or raw_symptoms:
            with st.spinner("Gemini AI is analyzing the clinical history..."):
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    
                    prompt = """
                    You are an expert Ayurvedic clinical AI. Analyze the patient's statement.
                    Extract the following:
                    1. The primary chief complaint.
                    2. A structured medical history summary.
                    3. The probable Dosha imbalance (Vikriti: Vata, Pitta, or Kapha) based on symptoms.
                    
                    Return ONLY a JSON object with the exact keys: "complaint", "history", and "dosha".
                    """
                    
                    contents = []
                    if audio_value:
                        audio_bytes = audio_value.read()
                        contents.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
                    if raw_symptoms:
                        contents.append(f"\nText Statement: {raw_symptoms}")
                        
                    contents.append(prompt)
                    
                    # Automated retry loop for handling 503 server overloads
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    temperature=0.1,
                                    response_mime_type="application/json"
                                )
                            )
                            break
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                time.sleep(3)
                            else:
                                raise e
                    
                    ai_data = json.loads(response.text)
                    
                    st.session_state['structured_complaint'] = ai_data.get("complaint", "Unknown")
                    st.session_state['structured_history'] = ai_data.get("history", "No history generated.")
                    st.session_state['dosha_imbalance'] = ai_data.get("dosha", "Unclear")
                    
                    urgent_keywords = ["chest pain", "severe", "shortness of breath", "chhati mein dard", "gambhīr"]
                    urgent = any(word in str(ai_data).lower() for word in urgent_keywords)
                    
                    save_patient_record(
                        st.session_state['current_patient'], 
                        st.session_state['current_age'], 
                        st.session_state['structured_complaint'], 
                        st.session_state['structured_history'], 
                        st.session_state['dosha_imbalance'],
                        urgent
                    )
                    
                    st.success(f"History processed! Detected Imbalance: {st.session_state['dosha_imbalance']}")
                    
                    # --- VOICE ASSISTANT CONFIRMATION ---
                    base_msg = f"Thank you, {st.session_state['current_patient']}. We have recorded your complaint of {st.session_state['structured_complaint']}. The doctor will see you shortly."
                    
                    lang_codes = {"Hindi": "hi", "English": "en", "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Marathi": "mr"}
                    target_code = lang_codes[patient_lang]
                    
                    if patient_lang != "English":
                        final_msg = translate_with_gemini(base_msg, patient_lang)
                    else:
                        final_msg = base_msg
                        
                    st.info(f"🗣️ AI Assistant: {final_msg}")
                    
                    tts = gTTS(text=final_msg, lang=target_code)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
                    
                except Exception as e:
                    st.error(f"Error processing connection: {e}")
        else:
            st.warning("Please record audio or type your symptoms.")

# --- TAB 3: UPLOAD REPORTS (Gemini Vision OCR) ---
with tab3:
    st.header("📄 Upload Previous Medical Documents")
    uploaded_file = st.file_uploader("Upload Lab Report or Prescription", type=["jpg", "jpeg", "png", "pdf", "webp"])
    
    if uploaded_file is not None:
        if "image" in uploaded_file.type:
            st.image(uploaded_file, caption="Uploaded Document", width=300)
        else:
            st.write(f"📄 {uploaded_file.name} uploaded.")
            
        if st.button("Extract Data (OCR)", key="btn_extract_ocr"):
            with st.spinner("Gemini Vision is scanning the document..."):
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    
                    doc_data = uploaded_file.getvalue()
                    doc_mime_type = uploaded_file.type
                    document_part = types.Part.from_bytes(data=doc_data, mime_type=doc_mime_type)
                    
                    prompt = """
                    You are a medical data extraction AI. Read this medical document and extract:
                    1. Patient Name & Date (if visible)
                    2. Diagnoses or Conditions
                    3. Prescribed Medications (with dosages)
                    4. Key Lab Results (flag any abnormal values)
                    
                    Format the output cleanly using markdown bullet points. If something is missing, just skip it. Do not invent information.
                    """
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model='gemini-3.6-flash',
                                contents=[document_part, prompt]
                            )
                            break
                        except Exception as e:
                            if "503" in str(e) and attempt < max_retries - 1:
                                time.sleep(3)
                            else:
                                raise e
                    
                    st.success("Document analyzed successfully!")
                    st.markdown("### 📋 Extracted Clinical Data")
                    st.info(response.text)
                    st.session_state['last_scanned_doc'] = response.text
                    
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# --- TAB 4: DOCTOR DASHBOARD ---
with tab4:
    st.header("👨‍⚕️ Physician Dashboard")
    
    conn = sqlite3.connect('medikiosk.db')
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Refresh Patient Queue", key="btn_refresh_dashboard"):
            st.rerun()
    
    if not df.empty:
        df['Triage Status'] = df['is_urgent'].apply(
            lambda x: "🚨 URGENT" if x == 1 else "🟢 Routine"
        )
        
        display_df = df[['id', 'name', 'age', 'Triage Status', 'dosha', 'complaint', 'history', 'date']]
        
        with col2:
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to Excel (CSV)",
                data=csv_data,
                file_name=f"hospital_records_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                key="btn_export_csv"
            )
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # --- PATIENT ACTIONS (Gemini Translate & Discharge) ---
        st.divider()
        st.subheader("Manage & Translate Records (Gemini AI)")
        
        patient_list = df.apply(lambda row: f"ID: {row['id']} - {row['name']}", axis=1).tolist()
        selected_patient = st.selectbox("Select a patient record to view/manage:", patient_list)
        
        pat_id = int(selected_patient.split(" - ")[0].replace("ID: ", ""))
        patient_data = df[df['id'] == pat_id].iloc[0]
        
        col_trans, col_del = st.columns([2, 1])
        
        with col_trans:
            st.write("**Translate Medical History:**")
            lang_options = ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi"]
            selected_lang = st.selectbox("Select Language", options=lang_options)
            
            if st.button("Translate History", key="btn_translate"):
                with st.spinner(f"Translating to {selected_lang}..."):
                    translated_text = translate_with_gemini(patient_data['history'], selected_lang)
                    st.success("Translation Complete")
                    st.info(translated_text)
                    
        with col_del:
            st.write("**Discharge Actions:**")
            if st.button("Confirm Discharge", type="primary", key="btn_confirm_discharge"):
                delete_patient(pat_id)
                st.success(f"Patient {pat_id} discharged successfully.")
                time.sleep(1)
                st.rerun()
                
    else:
        st.info("No patients in the queue.")