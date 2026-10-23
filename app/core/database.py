"""SQLite persistence: requests, feedback, evaluations, routing failures, and the job queue.

The API, the verification worker, and the dashboard all open the same file, so every call
uses a short-lived connection and the database runs in WAL mode (readers don't block writers).
"""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.core.config import DATA_DIR

# Verification states in which a request's final answer is not settled yet.
PENDING_VERIFICATION = frozenset({"pending", "running"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_hash TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    complexity TEXT NOT NULL,
    confidence REAL NOT NULL,
    route_reason TEXT NOT NULL,
    used_fallback INTEGER NOT NULL,
    output_text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    original_model_id TEXT NOT NULL DEFAULT '',
    request_type TEXT NOT NULL DEFAULT 'general',
    verification_status TEXT NOT NULL DEFAULT 'disabled',
    verification_model_id TEXT,
    quality_score REAL,
    quality_threshold REAL,
    quality_gap REAL,
    verification_reason TEXT NOT NULL DEFAULT '',
    verification_cost_usd REAL NOT NULL DEFAULT 0,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalation_model_id TEXT,
    escalation_cost_delta_usd REAL NOT NULL DEFAULT 0,
    initial_output_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    case_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER NOT NULL,
    model_cost_usd REAL NOT NULL DEFAULT 0,
    judge_cost_usd REAL NOT NULL DEFAULT 0,
    judge_score REAL,
    judge_model_id TEXT,
    judge_reason TEXT,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE REFERENCES requests(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    complexity TEXT NOT NULL,
    target_complexity TEXT NOT NULL,
    request_type TEXT NOT NULL,
    original_model_id TEXT NOT NULL,
    verifier_model_id TEXT NOT NULL,
    quality_score REAL NOT NULL,
    quality_threshold REAL NOT NULL,
    quality_gap REAL NOT NULL,
    reason TEXT NOT NULL,
    features_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verification_jobs (
    request_id TEXT PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
    max_output_tokens INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    queued_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    lease_until TEXT,
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_routing_failures_created ON routing_failures(created_at);
CREATE INDEX IF NOT EXISTS idx_verification_jobs_status ON verification_jobs(status, queued_at);
"""

# Columns introduced after the first release. Databases created by an older version are
# upgraded in place by `_migrate`; fresh databases already have them via `_SCHEMA`.
_ADDED_COLUMNS = {
    "requests": (
        ("prompt_hash", "TEXT NOT NULL DEFAULT ''"),
        ("original_model_id", "TEXT NOT NULL DEFAULT ''"),
        ("request_type", "TEXT NOT NULL DEFAULT 'general'"),
        ("verification_status", "TEXT NOT NULL DEFAULT 'disabled'"),
        ("verification_model_id", "TEXT"),
        ("quality_score", "REAL"),
        ("quality_threshold", "REAL"),
        ("quality_gap", "REAL"),
        ("verification_reason", "TEXT NOT NULL DEFAULT ''"),
        ("verification_cost_usd", "REAL NOT NULL DEFAULT 0"),
        ("escalated", "INTEGER NOT NULL DEFAULT 0"),
        ("escalation_model_id", "TEXT"),
        ("escalation_cost_delta_usd", "REAL NOT NULL DEFAULT 0"),
        ("initial_output_text", "TEXT NOT NULL DEFAULT ''"),
    ),
    "evaluations": (
        ("judge_score", "REAL"),
        ("judge_model_id", "TEXT"),
        ("judge_reason", "TEXT"),
        ("model_cost_usd", "REAL NOT NULL DEFAULT 0"),
        ("judge_cost_usd", "REAL NOT NULL DEFAULT 0"),
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("LLM_AUTOPILOT_DB") or DATA_DIR / "autopilot.sqlite3"
        self.path = str(configured)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self):
        """Open a connection, commit on success, roll back on error, always close."""
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def _fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(sql, params).fetchone()
        return dict(row) if row else None

    def initialize(self) -> None:
        with self._connection() as connection:
            if self.path != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            for column, sql_type in columns:
                if column not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

        # Backfill values that older rows could not have had.
        connection.execute(
            "UPDATE requests SET original_model_id = model_id WHERE original_model_id = ''"
        )
        connection.execute(
            "UPDATE requests SET initial_output_text = output_text WHERE initial_output_text = ''"
        )
        unhashed = connection.execute("SELECT id, prompt FROM requests WHERE prompt_hash = ''")
        connection.executemany(
            "UPDATE requests SET prompt_hash = ? WHERE id = ?",
            [(_sha256(row["prompt"]), row["id"]) for row in unhashed.fetchall()],
        )

    # ---- requests -----------------------------------------------------------------------

    def record_request(self, row: dict) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO requests
                (id, created_at, prompt, prompt_hash, model_id, provider_name, complexity,
                 confidence, route_reason, used_fallback, output_text, input_tokens,
                 output_tokens, latency_ms, cost_usd, original_model_id, request_type,
                 verification_status, initial_output_text)
                VALUES (:id, :created_at, :prompt, :prompt_hash, :model_id, :provider_name,
                        :complexity, :confidence, :route_reason, :used_fallback, :output_text,
                        :input_tokens, :output_tokens, :latency_ms, :cost_usd,
                        :original_model_id, :request_type, :verification_status,
                        :initial_output_text)""",
                {
                    **row,
                    "created_at": row.get("created_at", _now()),
                    "prompt_hash": row.get("prompt_hash") or _sha256(row["prompt"]),
                    "original_model_id": row.get("original_model_id", row["model_id"]),
                    "request_type": row.get("request_type", "general"),
                    "verification_status": row.get("verification_status", "disabled"),
                    "initial_output_text": row.get("initial_output_text", row["output_text"]),
                },
            )

    def update_verification(
        self,
        request_id: str,
        *,
        status: str,
        request_type: str,
        reason: str,
        verifier_model_id: str | None = None,
        quality_score: float | None = None,
        quality_threshold: float | None = None,
        quality_gap: float | None = None,
        verifier_response=None,
        verifier_model=None,
        original_cost_usd: float = 0.0,
        escalated: bool = False,
    ) -> None:
        """Record a verification step.

        When a verifier response is supplied its cost, tokens, and latency are folded into
        the request's totals. If ``escalated``, the verifier's answer also replaces the
        original one and the cost delta is recorded.
        """
        optional = {
            "verification_model_id": verifier_model_id,
            "quality_score": quality_score,
            "quality_threshold": quality_threshold,
            "quality_gap": quality_gap,
        }
        updates = {
            "verification_status": status,
            "request_type": request_type,
            "verification_reason": reason,
            **{column: value for column, value in optional.items() if value is not None},
        }
        assignments: list[str] = []
        increments: list[float] = []
        if verifier_response is not None:
            updates["verification_cost_usd"] = round(verifier_response.cost_usd, 10)
            if escalated:
                updates.update(
                    model_id=verifier_response.model_id,
                    provider_name=verifier_model.provider_name,
                    output_text=verifier_response.output_text,
                    escalated=1,
                    escalation_model_id=verifier_response.model_id,
                    escalation_cost_delta_usd=round(
                        verifier_response.cost_usd - original_cost_usd, 10
                    ),
                )
            assignments = [
                "cost_usd = ROUND(cost_usd + ?, 10)",
                "input_tokens = input_tokens + ?",
                "output_tokens = output_tokens + ?",
                "latency_ms = latency_ms + ?",
            ]
            increments = [
                verifier_response.cost_usd,
                verifier_response.input_tokens,
                verifier_response.output_tokens,
                verifier_response.latency_ms,
            ]
        set_clause = ", ".join([*(f"{column} = ?" for column in updates), *assignments])
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE requests SET {set_clause} WHERE id = ?",
                [*updates.values(), *increments, request_id],
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown request_id: {request_id}")

    def recent_requests(self, limit: int = 50) -> list[dict]:
        return self._fetch_all(
            """SELECT r.*, f.rating AS feedback_rating FROM requests r
               LEFT JOIN feedback f ON f.request_id = r.id
               ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        )

    def request_by_id(self, request_id: str) -> dict | None:
        return self._fetch_one(
            """SELECT r.*, f.rating AS feedback_rating FROM requests r
               LEFT JOIN feedback f ON f.request_id = r.id WHERE r.id = ?""",
            (request_id,),
        )

    def cost_dashboard_requests(self) -> list[dict]:
        """Inference telemetry for cost and routing charts (no prompt text)."""
        return self._fetch_all(
            """SELECT id, created_at, prompt_hash, original_model_id, model_id,
                      complexity, input_tokens, output_tokens, latency_ms, cost_usd,
                      quality_score, escalated, verification_status
               FROM requests ORDER BY created_at"""
        )

    def cost_usage_summary(self) -> dict:
        """Inference usage only; golden-set runs are excluded and no model is called."""
        row = self._fetch_one(
            """SELECT COUNT(*) AS request_count,
                      COALESCE(SUM(input_tokens), 0) AS input_tokens,
                      COALESCE(SUM(output_tokens), 0) AS output_tokens,
                      COALESCE(SUM(cost_usd), 0) AS actual_cost_usd
               FROM requests"""
        )
        return {
            "request_count": int(row["request_count"]),
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "actual_cost_usd": round(float(row["actual_cost_usd"]), 10),
        }

    # ---- feedback and evaluations -------------------------------------------------------

    def add_feedback(self, request_id: str, rating: int, comment: str = "") -> bool:
        """Save (or overwrite) a rating; returns False when the request does not exist."""
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM requests WHERE id = ?", (request_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                """INSERT INTO feedback(request_id, rating, comment, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(request_id) DO UPDATE SET rating = excluded.rating,
                   comment = excluded.comment, created_at = excluded.created_at""",
                (request_id, rating, comment, _now()),
            )
        return True

    def record_evaluation_run(self, run_id: str, rows: list[dict]) -> None:
        now = _now()
        with self._connection() as connection:
            connection.executemany(
                """INSERT INTO evaluations
                (run_id, created_at, case_id, model_id, score, passed, judge_score,
                 judge_model_id, judge_reason, model_cost_usd, judge_cost_usd, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        now,
                        row["case_id"],
                        row["model_id"],
                        row["score"],
                        int(row["passed"]),
                        row.get("judge_score"),
                        row.get("judge_model_id"),
                        row.get("judge_reason"),
                        row.get("model_cost_usd", 0.0),
                        row.get("judge_cost_usd", 0.0),
                        json.dumps(
                            {
                                "checks": row["checks"],
                                "judge": row.get("judge"),
                                "output_text": row.get("output_text", ""),
                                "input_tokens": row.get("input_tokens", 0),
                                "output_tokens": row.get("output_tokens", 0),
                                "latency_ms": row.get("latency_ms", 0.0),
                            },
                            ensure_ascii=False,
                        ),
                    )
                    for row in rows
                ],
            )

    def dashboard_evaluations(self) -> list[dict]:
        return self._fetch_all(
            "SELECT created_at, score, passed, model_id FROM evaluations ORDER BY created_at"
        )

    def evaluation_scores(self, limit: int = 100) -> list[float]:
        """Newest-first scores, the order ``detect_drift`` expects."""
        rows = self._fetch_all("SELECT score FROM evaluations ORDER BY id DESC LIMIT ?", (limit,))
        return [float(row["score"]) for row in rows]

    # ---- routing failures ---------------------------------------------------------------

    def record_routing_failure(self, row: dict) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO routing_failures
                (request_id, created_at, prompt, complexity, target_complexity, request_type,
                 original_model_id, verifier_model_id, quality_score, quality_threshold,
                 quality_gap, reason, features_json)
                VALUES (:request_id, :created_at, :prompt, :complexity, :target_complexity,
                        :request_type, :original_model_id, :verifier_model_id, :quality_score,
                        :quality_threshold, :quality_gap, :reason, :features_json)""",
                {**row, "created_at": row.get("created_at", _now())},
            )

    def training_failures(self) -> list[dict]:
        """Routing failures relabeled with the tier they needed, for classifier retraining."""
        return self._fetch_all(
            """SELECT prompt, target_complexity AS label, request_type, quality_gap,
                      original_model_id, verifier_model_id
               FROM routing_failures ORDER BY id"""
        )

    def recent_routing_failures(self, limit: int = 50) -> list[dict]:
        return self._fetch_all(
            """SELECT f.*, r.escalation_model_id, r.escalation_cost_delta_usd,
                      r.verification_cost_usd
               FROM routing_failures f JOIN requests r ON r.id = f.request_id
               ORDER BY f.id DESC LIMIT ?""",
            (limit,),
        )

    # ---- verification queue -------------------------------------------------------------

    def enqueue_verification_job(self, request_id: str, max_output_tokens: int) -> None:
        now = _now()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO verification_jobs
                   (request_id, max_output_tokens, status, queued_at, updated_at)
                   VALUES (?, ?, 'queued', ?, ?)
                   ON CONFLICT(request_id) DO NOTHING""",
                (request_id, max_output_tokens, now, now),
            )

    def claim_verification_job(self, lease_seconds: int = 300) -> dict | None:
        """Atomically lease the oldest queued job, reclaiming leases that expired."""
        now = datetime.now(UTC)
        now_text = now.isoformat()
        lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")  # take the write lock before selecting
            row = connection.execute(
                """SELECT request_id, max_output_tokens, attempts
                   FROM verification_jobs
                   WHERE status = 'queued'
                      OR (status = 'running' AND (lease_until IS NULL OR lease_until <= ?))
                   ORDER BY queued_at LIMIT 1""",
                (now_text,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE verification_jobs
                   SET status = 'running', attempts = attempts + 1,
                       updated_at = ?, lease_until = ?, last_error = ''
                   WHERE request_id = ?""",
                (now_text, lease_until, row["request_id"]),
            )
            return {
                "request_id": row["request_id"],
                "max_output_tokens": int(row["max_output_tokens"]),
                "attempts": int(row["attempts"]) + 1,
            }

    def finish_verification_job(self, request_id: str, *, error: str | None = None) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE verification_jobs
                   SET status = ?, updated_at = ?, lease_until = NULL, last_error = ?
                   WHERE request_id = ?""",
                ("failed" if error else "completed", _now(), error or "", request_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown verification job: {request_id}")

    def verification_job(self, request_id: str) -> dict | None:
        return self._fetch_one(
            "SELECT * FROM verification_jobs WHERE request_id = ?", (request_id,)
        )

    # ---- aggregate metrics --------------------------------------------------------------

    def metrics(self) -> dict:
        with self._connection() as connection:
            requests = connection.execute(
                """SELECT COUNT(*) AS request_count,
                          COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                          COALESCE(AVG(latency_ms), 0) AS average_latency_ms,
                          COALESCE(AVG(input_tokens + output_tokens), 0) AS average_tokens,
                          COALESCE(SUM(escalated = 1), 0) AS escalation_count,
                          COALESCE(SUM(verification_cost_usd), 0) AS verification_cost,
                          COALESCE(SUM(escalation_cost_delta_usd), 0) AS escalation_delta,
                          COALESCE(SUM(verification_status IN ('pending', 'running')), 0)
                              AS pending_count
                   FROM requests"""
            ).fetchone()
            feedback = connection.execute(
                "SELECT COUNT(*) AS count, COALESCE(AVG(rating), 0) AS average FROM feedback"
            ).fetchone()
            evaluations = connection.execute(
                """SELECT COUNT(*) AS count,
                          COALESCE(AVG(score), 0) AS average,
                          COUNT(judge_score) AS judge_count,
                          COALESCE(AVG(judge_score), 0) AS judge_average,
                          COALESCE(SUM(model_cost_usd), 0) AS model_cost,
                          COALESCE(SUM(judge_cost_usd), 0) AS judge_cost
                   FROM evaluations"""
            ).fetchone()
            failure_count = connection.execute(
                "SELECT COUNT(*) AS count FROM routing_failures"
            ).fetchone()["count"]
        return {
            "request_count": int(requests["request_count"]),
            "total_cost_usd": round(
                float(requests["total_cost_usd"])
                + float(evaluations["model_cost"])
                + float(evaluations["judge_cost"]),
                10,
            ),
            "evaluation_cost_usd": round(float(evaluations["model_cost"]), 10),
            "judge_cost_usd": round(float(evaluations["judge_cost"]), 10),
            "average_latency_ms": round(float(requests["average_latency_ms"]), 3),
            "average_tokens": round(float(requests["average_tokens"]), 2),
            "feedback_count": int(feedback["count"]),
            "average_feedback_rating": round(float(feedback["average"]), 2),
            "evaluation_count": int(evaluations["count"]),
            "average_evaluation_score": round(float(evaluations["average"]), 4),
            "judge_evaluation_count": int(evaluations["judge_count"]),
            "average_judge_score": round(float(evaluations["judge_average"]), 4),
            "routing_failure_count": int(failure_count),
            "escalation_count": int(requests["escalation_count"]),
            "verification_pending_count": int(requests["pending_count"]),
            "verification_cost_usd": round(float(requests["verification_cost"]), 10),
            "escalation_cost_delta_usd": round(float(requests["escalation_delta"]), 10),
        }
