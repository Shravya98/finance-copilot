import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum

from app.agents.agent import FinanceAgent
from app.memory.context_buffer import SessionStore


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    session_id: str
    question: str
    status: JobStatus = JobStatus.PENDING
    answer: str | None = None
    error: str | None = None


class JobManager:
    """Runs agent calls in the background so clients can poll for results."""

    def __init__(self, agent: FinanceAgent, sessions: SessionStore):
        self.agent = agent
        self.sessions = sessions
        self.jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def submit(self, session_id: str, question: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex, session_id=session_id, question=question)
        self.jobs[job.job_id] = job

        task = asyncio.create_task(self._run(job))   # start in the background
        self._tasks.add(task)                        # keep a reference
        task.add_done_callback(self._tasks.discard)   # clean up when finished
        return job

    async def _run(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        buffer = self.sessions.get(job.session_id)
        try:
            answer = await self.agent.run(buffer.get_messages(), job.question)
            buffer.add_user_message(job.question)
            buffer.add_ai_message(answer)
            job.answer = answer
            job.status = JobStatus.COMPLETED
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.FAILED

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)