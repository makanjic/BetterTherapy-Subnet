from .session import session
from .models import Request, MinerResponse
from datetime import datetime, timedelta, timezone
import typing
from sqlalchemy.orm import Session, selectinload


# get request with greater than or equal to 24 hours old
@session
def get_ready_requests(session: Session, hours: int = 24) -> typing.List[Request]:
    """Get requests that are older than the specified number of hours."""

    threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
    # attach responses to the requests
    return (
        session.query(Request)
        .filter(Request.created_at < threshold)
        .options(selectinload(Request.responses))
        .all()
    )


@session
def add_request(
    session: Session, name: str, openai_batch_id: str, prompt: str, base_response: str
) -> Request:
    """Add a new request to the database."""
    new_request = Request(
        name=name,
        openai_batch_id=openai_batch_id,
        prompt=prompt,
        base_response=base_response,
    )
    session.add(new_request)
    session.commit()
    session.refresh(new_request)
    return new_request


@session
def add_bulk_responses(session: Session, responses: typing.List[MinerResponse]) -> None:
    """Add multiple responses to a request."""
    session.bulk_save_objects(responses)
    session.commit()


@session
def delete_requests(session: Session, request_ids: typing.List[int]) -> None:
    """Delete requests by their IDs."""
    session.query(Request).filter(Request.id.in_(request_ids)).delete(
        synchronize_session=False
    )
    session.commit()
