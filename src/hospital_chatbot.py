"""
PURPOSE:
An interactive, domain-specific Clinical Assistant Softbot designed for hospital 
operations. It allows clinical staff (doctors, nurses, administrative triage) to 
query live bed capacity, emergency escalation protocols, 30-day readmission 
guidelines, and discharge policies via natural language.
"""

import re

# Domain Knowledge Base for Hospital Operations
KNOWLEDGE_BASE = {
    "beds": (
        "Current Capacity Report:\n"
        "  - General Medical Ward 3A: 2 beds available.\n"
        "  - Surgical Ward 2B: 4 beds available.\n"
        "  - Intensive Care Unit (ICU): 95% capacity (1 emergency bed open).\n"
        "  - Emergency Casualty: 6 bays occupied, 2 bays available."
    ),
    "protocol": (
        "Emergency Escalation Protocol:\n"
        "  - Code Blue (Cardiac/Resp Arrest): Dial ext 2222 immediately for Resus Team.\n"
        "  - Code Red (Fire/Hazard): Dial ext 1111 and initiate zone evacuation.\n"
        "  - Sepsis Alert: Follow 1-hour bundle (lactate, blood cultures, broad antibiotics)."
    ),
    "readmission": (
        "30-Day Readmission Mitigation Policy:\n"
        "  - Patients flagged with >40% readmission risk require pharmacy discharge consultation.\n"
        "  - Mandatory 48-hour post-discharge telephonic check-in by community health team.\n"
        "  - Coordinate 7-day follow-up slot at local municipal outpatient clinic."
    ),
    "visiting": (
        "Hospital Visiting Regulations:\n"
        "  - General Wards: 14:00 - 16:00 and 18:00 - 19:30 daily.\n"
        "  - Intensive Care Unit (ICU): Immediate family only; 15:00 - 16:00 (Max 2 visitors).\n"
        "  - Pediatric Ward: One parent permitted 24/7 bedside stay."
    ),
    "discharge": (
        "Clinical Discharge Procedure:\n"
        "  - Final medication reconciliation must be signed off by attending physician.\n"
        "  - Discharge summary must be submitted to the Electronic Health Record (EHR).\n"
        "  - Patient transfer forms must be given to transport if ambulance is required."
    ),
    "triage": (
        "South African Triage Scale (SATS) Categories:\n"
        "  - Red (Emergency): Immediate resuscitation needed.\n"
        "  - Orange (Very Urgent): Seen within 10 minutes.\n"
        "  - Yellow (Urgent): Seen within 1 hour.\n"
        "  - Green (Non-urgent): Seen within 4 hours."
    )
}

def triage_assistant_response(user_input: str) -> str:
    """Classifies user intent and returns relevant operational guidance."""
    cleaned = user_input.lower().strip()
    
    # 1. Check for exit commands
    if cleaned in ["exit", "quit", "bye", "stop"]:
        return "Session terminated. Thank you for using Triage AI. Have a safe shift."
    
    # 2. Match intents based on clinical keywords
    if any(k in cleaned for k in ["bed", "capacity", "ward", "occupancy", "space"]):
        return KNOWLEDGE_BASE["beds"]
    
    elif any(k in cleaned for k in ["protocol", "code blue", "code red", "emergency", "crisis", "arrest"]):
        return KNOWLEDGE_BASE["protocol"]
    
    elif any(k in cleaned for k in ["readmission", "bounce back", "risk", "follow-up", "follow up"]):
        return KNOWLEDGE_BASE["readmission"]
    
    elif any(k in cleaned for k in ["visit", "visiting", "hours", "family"]):
        return KNOWLEDGE_BASE["visiting"]
    
    elif any(k in cleaned for k in ["discharge", "leaving", "checkout", "release"]):
        return KNOWLEDGE_BASE["discharge"]
        
    elif any(k in cleaned for k in ["triage", "sats", "priority", "urgency", "red", "orange"]):
        return KNOWLEDGE_BASE["triage"]
    
    # 3. Fallback for unrecognized questions
    else:
        return (
            "I could not identify that specific operational query. "
            "You can ask me about:\n"
            "  - Bed capacity / Ward occupancy\n"
            "  - Emergency escalation protocols (Code Blue, Sepsis)\n"
            "  - Readmission mitigation rules\n"
            "  - Discharge procedures\n"
            "  - Visiting hours\n"
            "  - Triage priority levels (SATS)"
        )

def run_chatbot_interactive():
    """Launches the interactive command-line session."""
    print("=" * 70)
    print("      TRIAGE AI: CLINICAL & OPERATIONAL ASSISTANT SOFTBOT")
    print("=" * 70)
    print("System active. Type your question below, or type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("Staff Member > ")
            if not user_input.strip():
                continue
                
            response = triage_assistant_response(user_input)
            print(f"\nTriage-Bot:\n{response}\n")
            print("-" * 70)
            
            if user_input.lower().strip() in ["exit", "quit", "bye", "stop"]:
                break
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

if __name__ == "__main__":
    run_chatbot_interactive()