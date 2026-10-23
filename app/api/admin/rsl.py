from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel, Field

from app.api.admin._auth import require_admin
from app.core.routing_policy import load_routing_policy
from app.core.rsl import classify, select_model
from app.core.rsl_runtime import candidate_costs
from app.schemas.chat import ChatMessage

router = APIRouter(prefix="/rsl", dependencies=[Depends(require_admin)])


class Feedback(BaseModel):
    attempt_id: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(default="", max_length=1000)


class Preview(BaseModel):
    routing_profile_id: str = ""
    messages: list[ChatMessage]
    allowed_models: list[str] | None = None
    explore: bool = True


@router.get("/stats")
async def stats(request: Request, routing_profile_id: str = ""):
    return {"data": request.app.state.rsl_store.stats(routing_profile_id)}


@router.get("/feedback")
async def feedback(request: Request, routing_profile_id: str = "", limit: int = Query(100, ge=1, le=1000)):
    return {"data": request.app.state.rsl_store.feedback(routing_profile_id, limit)}


@router.post("/feedback")
async def submit(body: Feedback, request: Request):
    try:
        request.app.state.rsl_store.correct(body.attempt_id, body.score, body.reason)
    except KeyError:
        raise HTTPException(404, "Unknown attempt_id")
    return {"ok": True, "attempt_id": body.attempt_id}


@router.post("/preview")
async def preview(body: Preview, request: Request):
    policy = await load_routing_policy(request.app.state.db, body.routing_profile_id)
    if not policy.rsl_enabled:
        raise HTTPException(400, "RSL is disabled for this profile")
    engine = request.app.state.router
    allowed = policy.effective_allowed_models(set(body.allowed_models) if body.allowed_models is not None else None,
                                               set(engine.model_map))
    candidates = candidate_costs(engine, allowed, body.messages, policy.recommendation_excluded_models)
    if not candidates:
        raise HTTPException(503, "No allowed, available RSL models")
    return select_model(request.app.state.rsl_store, candidates, policy, body.routing_profile_id,
                        classify(body.messages), explore=body.explore)
