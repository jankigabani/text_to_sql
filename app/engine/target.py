"""Which database a request talks to.

The UI sends connection details with every question, so the same running server can
answer questions about any PostgreSQL database the user points it at. A `DBTarget` is
that set of details, plus a fingerprint used to cache pools and catalogs per database.

The password is a `SecretStr`: it never appears in logs, in `repr`, in the run log, or in
any response sent back to the browser.
"""
from __future__ import annotations

import hashlib
from typing import Literal

from psycopg.conninfo import make_conninfo
from pydantic import BaseModel, Field, SecretStr

from app.core.config import get_settings


class DBTarget(BaseModel):
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    database: str = Field(min_length=1)
    user: str = Field(min_length=1)
    password: SecretStr = SecretStr("")
    sslmode: Literal["disable", "allow", "prefer", "require"] = "prefer"
    schemas: list[str] | None = None
    include_views: bool = False

    def conninfo(self, statement_timeout_ms: int | None = None) -> str:
        s = get_settings()
        timeout = statement_timeout_ms or s.gate_statement_timeout_ms
        options = f"-c default_transaction_read_only=on -c statement_timeout={timeout}"
        return make_conninfo(
            host=self.host, port=self.port, dbname=self.database, user=self.user,
            password=self.password.get_secret_value(), sslmode=self.sslmode,
            connect_timeout=8, application_name="t2s-gate", options=options,
        )

    def fingerprint(self) -> str:
        pw = hashlib.sha256(self.password.get_secret_value().encode()).hexdigest()[:12]
        key = "|".join([
            self.host.lower(), str(self.port), self.database, self.user, pw,
            ",".join(sorted(self.schemas)) if self.schemas else "*",
            "v" if self.include_views else "t",
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def public(self) -> dict:
        """Safe to log and to return to the browser."""
        return {
            "host": self.host, "port": self.port, "database": self.database,
            "user": self.user, "sslmode": self.sslmode, "schemas": self.schemas,
            "include_views": self.include_views,
        }

    @classmethod
    def from_settings(cls, database: str | None = None) -> "DBTarget":
        s = get_settings()
        return cls(
            host=s.pghost, port=s.pgport, database=database or s.pgdatabase,
            user=s.pguser, password=SecretStr(s.pgpassword),
        )


class ConnectionIn(BaseModel):
    """What the browser sends. `use_default` means 'the database in .env', so the
    password for the default database never has to travel to the browser at all."""
    use_default: bool = False
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    sslmode: Literal["disable", "allow", "prefer", "require"] = "prefer"
    schemas: list[str] | None = None
    include_views: bool = False

    def to_target(self) -> DBTarget:
        if self.use_default:
            base = DBTarget.from_settings(self.database or None)
            return base.model_copy(update={
                "schemas": self.schemas, "include_views": self.include_views,
            })
        if not self.database or not self.user:
            raise ValueError("database and user are required")
        return DBTarget(
            host=self.host or "localhost", port=self.port, database=self.database,
            user=self.user, password=SecretStr(self.password), sslmode=self.sslmode,
            schemas=self.schemas or None, include_views=self.include_views,
        )
