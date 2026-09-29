"""
Human-in-the-Loop (HITL) approval manager for Text2SQL.

Ensures that every generated SQL query is presented to the user
and requires explicit approval before database execution.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

STATUS_PENDING = "PENDING_APPROVAL"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"


@dataclass
class ApprovalRequest:
    """A generated SQL query awaiting human inspection and approval."""
    query_id: str
    question: str
    generated_sql: str
    explanation: str
    status: str = STATUS_PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None


# Thread-safe in-memory store for approval requests
_APPROVAL_REGISTRY: Dict[str, ApprovalRequest] = {}


def create_approval_request(question: str, sql: str, explanation: str) -> ApprovalRequest:
    """Register a newly generated SQL query requiring human approval."""
    query_id = str(uuid.uuid4())
    req = ApprovalRequest(
        query_id=query_id,
        question=question,
        generated_sql=sql,
        explanation=explanation,
        status=STATUS_PENDING,
    )
    _APPROVAL_REGISTRY[query_id] = req
    return req


def get_approval_request(query_id: str) -> Optional[ApprovalRequest]:
    """Retrieve an approval request by its unique query_id."""
    return _APPROVAL_REGISTRY.get(query_id)


def approve_request(query_id: str) -> Tuple[bool, Optional[ApprovalRequest], str]:
    """
    Approve a pending SQL query.
    Returns: (success, request_object, message)
    """
    req = _APPROVAL_REGISTRY.get(query_id)
    if not req:
        return False, None, f"Query ID '{query_id}' not found."

    if req.status == STATUS_APPROVED:
        return True, req, "Query was already approved."

    if req.status == STATUS_REJECTED:
        return False, req, "Query was previously rejected."

    req.status = STATUS_APPROVED
    req.approved_at = datetime.now(timezone.utc)
    return True, req, "Query approved successfully."


def reject_request(query_id: str) -> Tuple[bool, Optional[ApprovalRequest], str]:
    """
    Reject a pending SQL query.
    Returns: (success, request_object, message)
    """
    req = _APPROVAL_REGISTRY.get(query_id)
    if not req:
        return False, None, f"Query ID '{query_id}' not found."

    req.status = STATUS_REJECTED
    return True, req, "Query rejected by user."
