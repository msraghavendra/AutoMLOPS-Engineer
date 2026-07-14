from sqlalchemy.orm import Session
from app.models import AuditLog

def log_event(db: Session, event_type: str, message: str, severity: str = "INFO") -> AuditLog:
    """
    Log an MLOps lifecycle event to the database.
    event_type: "TRAINING", "DEPLOYMENT", "DRIFT", "RETRAINING", "SYSTEM"
    severity: "INFO", "WARNING", "ERROR"
    """
    db_log = AuditLog(
        event_type=event_type,
        message=message,
        severity=severity
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log
