from flows.training_flow import task_archive_logs
from src.configs.loader import load_config

# Wir nutzen deine bestehende Config, um den Bucket-Namen dynamisch zu holen
ENV_CFG = load_config()
bucket = ENV_CFG["bucket_name"] 

if __name__ == "__main__":
    print(f"🚀 Starte Archivierungs-Test für Bucket: {bucket}...")
    
    # Wir rufen den Task direkt auf (.fn umgeht die Prefect-Orchestrierung für den Test)
    count = task_archive_logs.fn()
    
    
    if count > 0:
        print(f"✅ Erfolg! {count} Dateien wurden verschoben.")
    else:
        print("ℹ️ Keine Dateien zum Verschieben gefunden oder Fehler aufgetreten.")