import sqlite3
import os
import json
import time
import pickle
import numpy as np

DB_PATH = "clinical_workstation.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables and seeds the model registry."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        modality TEXT,
        scan_date TEXT
    )
    """)
    
    # 2. Scans/Audit table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        file_path TEXT,
        snr REAL,
        entropy REAL,
        privacy_score REAL,
        status TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    """)
    
    # 3. Clinical Audit Logs table (HIPAA Requirement)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_action TEXT,
        patient_id TEXT,
        details TEXT
    )
    """)
    
    # 4. Model Registry table (Model Versioning)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        model_name TEXT,
        version TEXT,
        status TEXT,
        accuracy REAL,
        f1_score REAL,
        registered_at TEXT,
        PRIMARY KEY (model_name, version)
    )
    """)
    
    # 5. Persistent Segmentation Cache table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS segmentation_cache (
        cache_key TEXT PRIMARY KEY,
        mask_blob BLOB,
        created_at REAL
    )
    """)
    
    conn.commit()
    
    # Seed model registry with default models if empty
    cursor.execute("SELECT COUNT(*) FROM model_registry")
    if cursor.fetchone()[0] == 0:
        default_models = [
            ("U-Net", "v1.0.0", "Production", 0.88, 0.87),
            ("U-Net", "v1.1.0-beta", "Staging", 0.90, 0.89),
            ("Attention U-Net", "v2.0.0", "Production", 0.92, 0.91),
            ("Attention U-Net", "v2.1.0-rc1", "Staging", 0.93, 0.92),
            ("U-Net++", "v1.0.0", "Production", 0.91, 0.90),
            ("Mask R-CNN", "v3.0.0", "Production", 0.88, 0.88),
            ("ResNet50", "v1.0.0", "Production", 0.89, 0.88),
            ("ResNet50", "v2.0.0", "Staging", 0.94, 0.93)
        ]
        cursor.executemany("""
        INSERT INTO model_registry (model_name, version, status, accuracy, f1_score, registered_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, default_models)
        conn.commit()
        
    conn.close()

def log_audit_action(user_action, patient_id, details=""):
    """Logs a clinical action to the audit logs database for compliance."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO audit_logs (timestamp, user_action, patient_id, details)
    VALUES (datetime('now'), ?, ?, ?)
    """, (user_action, patient_id, details))
    conn.commit()
    conn.close()

def save_patient(patient_id, age, gender, modality, scan_date=None):
    """Saves or updates a patient record."""
    if not scan_date:
        scan_date = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO patients (patient_id, age, gender, modality, scan_date)
    VALUES (?, ?, ?, ?, ?)
    """, (patient_id, age, gender, modality, scan_date))
    conn.commit()
    conn.close()
    log_audit_action("UPSERT_PATIENT", patient_id, f"Age: {age}, Gender: {gender}, Modality: {modality}")

def save_scan_metrics(patient_id, file_path, snr, entropy, privacy_score, status="Processed"):
    """Saves or updates metrics for a specific scan."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO scans (patient_id, file_path, snr, entropy, privacy_score, status)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, file_path, snr, entropy, privacy_score, status))
    conn.commit()
    conn.close()
    log_audit_action("SAVE_SCAN_METRICS", patient_id, f"SNR: {snr:.2f}, Entropy: {entropy:.2f}, Privacy Score: {privacy_score:.1f}%")

def get_patient_record(patient_id):
    """Retrieves patient details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_audit_logs(limit=50):
    """Retrieves the recent audit logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_model_registry():
    """Retrieves all registered models."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_registry ORDER BY model_name, version")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def compute_cache_key(image, model_name):
    """Generates a key for caching based on model name and image hash."""
    import hashlib
    img_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return f"{model_name}_{img_hash}"

def get_cached_segmentation(cache_key):
    """Retrieves a cached segmentation mask if it exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT mask_blob FROM segmentation_cache WHERE cache_key = ?", (cache_key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return pickle.loads(row['mask_blob'])
    return None

def cache_segmentation(cache_key, mask):
    """Saves a segmentation mask to the persistent cache."""
    conn = get_db_connection()
    cursor = conn.cursor()
    mask_blob = pickle.dumps(mask)
    cursor.execute("""
    INSERT OR REPLACE INTO segmentation_cache (cache_key, mask_blob, created_at)
    VALUES (?, ?, ?)
    """, (cache_key, mask_blob, time.time()))
    conn.commit()
    conn.close()
