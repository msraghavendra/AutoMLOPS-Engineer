from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models import AuditLog
from app.schemas import AuditLogRead

router = APIRouter(prefix="/api/logs", tags=["Audit Logs"])

@router.get("", response_model=List[AuditLogRead])
def list_logs(
    event_type: Optional[str] = Query(None, description="Filter logs by event type"),
    severity: Optional[str] = Query(None, description="Filter logs by severity level (INFO, WARNING, ERROR)"),
    limit: int = Query(50, ge=1, le=200, description="Number of logs to retrieve (pagination limit)"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Fetch system and MLOps audit logs with pagination and optional filters.
    """
    query = db.query(AuditLog)
    
    if event_type:
        query = query.filter(AuditLog.event_type == event_type)
    if severity:
        query = query.filter(AuditLog.severity == severity)
        
    return query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
