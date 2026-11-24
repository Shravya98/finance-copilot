
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.agent import FinanceAgent
from app.config import Settings
from app.models.schemas import ChatResponse
from app.services.llm_service import LLM
from app.services.vector_store_service import VectorStoreService
from app.memory.context_buffer import SessionStore
from app.scheduler.job_manager import JobManager
from app.tools.tools import FinanceTools

app = FastAPI(title="Finance Copilot")

# Initialize services
settings = Settings()
vector_store = VectorStoreService(settings)
vector_store.seed_if_empty()

llm = LLM(settings, vector_store)
finance_tools = FinanceTools(vector_store).get_tools()
llm.tool_chain(finance_tools)
app.state.llm_service = llm
agent = FinanceAgent(llm, finance_tools)
app.state.agent = agent
sessions = SessionStore()
job_manager = JobManager(agent, sessions)
app.state.job_manager = job_manager


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.get("/")
def home():
    return {"message": "Finance Copilot API is running"}


@app.post("/chat")
async def chat(request: ChatRequest):
    buffer = sessions.get(request.session_id)
    buffer.add_user_message(request.question)

    answer = await llm.ask(buffer.get_messages())

    buffer.add_ai_message(answer)
    return {"answer": answer}


@app.post("/chat/rag")
async def chat_rag(request: ChatRequest):
    buffer = sessions.get(request.session_id)
    buffer.add_user_message(request.question)

    answer = await llm.ask(buffer.get_messages())

    buffer.add_ai_message(answer)
    return {"answer": answer}


@app.post("/chat/structured")
async def chat_structured(request: ChatRequest):
    buffer = sessions.get(request.session_id)
    buffer.add_user_message(request.question)

    answer = await llm.ask_structured_output(
        buffer.get_messages()
    )

    buffer.add_ai_message(answer.model_dump_json())
    return answer


@app.post("/chat/tools", response_model=ChatResponse)
async def chat_with_tools(request: ChatRequest):
    try:
        buffer = sessions.get(request.session_id)
        buffer.add_user_message(request.question)

        llm_service = getattr(app.state, "llm_service", llm)
        answer = await llm_service.ask_with_tools(buffer.get_messages())

        buffer.add_ai_message(answer)
        return ChatResponse(question=request.question, answer=answer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {exc}",
        ) from exc


@app.post("/chat/agent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        buffer = sessions.get(request.session_id)
        history = buffer.get_messages()

        agent_instance = getattr(app.state, "agent", None)
        if agent_instance is None:
            raise ValueError("Finance agent is not initialized.")

        answer = await agent_instance.run(history, request.question)

        buffer.add_user_message(request.question)
        buffer.add_ai_message(answer)
        return ChatResponse(question=request.question, answer=answer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question with agent: {exc}",
        ) from exc


@app.post("/jobs")
async def submit_job(request: ChatRequest):
    try:
        manager = getattr(app.state, "job_manager", None)
        if manager is None:
            raise ValueError("Job manager is not initialized.")

        job = manager.submit(request.session_id, request.question)
        return {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "status": job.status.value,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit job: {exc}",
        ) from exc


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    try:
        manager = getattr(app.state, "job_manager", None)
        if manager is None:
            raise ValueError("Job manager is not initialized.")

        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        return {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "status": job.status.value,
            "answer": job.answer,
            "error": job.error,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job status: {exc}",
        ) from exc