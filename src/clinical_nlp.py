"""
Triage AI - Clinical NLP & Speech Synthesis (GitHub Issue #5)
Extracts structured medical entities from free-form doctor dictation 
and verbalizes critical triage alerts using text-to-speech synthesis.
"""
import re
import pyttsx3

# Sample unstructured dictation transcribed from a doctor's voice note
clinical_dictation = (
    "Patient is a 64-year-old male presenting with acute dyspnea, severe chest tightness, "
    "and peripheral edema. Oxygen saturation measured at 90 percent. Administered Furosemide "
    "40mg intravenously and prescribed Metformin 500mg daily. Recommend transfer to Ward 3B."
)

print("--- 1. Raw Clinical Dictation ---")
print(clinical_dictation)

print("\n--- 2. Parsing Medical Entities (NLP Pipeline) ---")
# Keyword and regex entity extraction
symptom_patterns = ["dyspnea", "chest tightness", "edema", "fever", "cough"]
medication_pattern = r"([A-Z][a-z]+)\s+(\d+\s*(?:mg|ml|percent))"

found_symptoms = [s for s in symptom_patterns if s in clinical_dictation.lower()]
found_medications = re.findall(medication_pattern, clinical_dictation)

extracted_data = {
    "Identified Symptoms": found_symptoms,
    "Medications & Dosages": found_medications,
    "Assigned Disposition": "Ward 3B"
}

for category, values in extracted_data.items():
    print(f"  {category}: {values}")

print("\n--- 3. Speech Synthesis: Verbalizing Triage Summary ---")
alert_summary = (
    f"Triage Alert: Patient diagnosed with {', '.join(found_symptoms)}. "
    f"Transferred to Ward 3B."
)
print(f"Speech Engine Speaking: '{alert_summary}'")

try:
    engine = pyttsx3.init()
    engine.say(alert_summary)
    engine.runAndWait()
    print("[SUCCESS] Audio synthesis complete.")
except Exception as e:
    print(f"[Note] Audio playback skipped (headless environment): {e}")