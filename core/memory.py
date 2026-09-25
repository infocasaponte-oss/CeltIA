# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import json
import re
import sqlite3
from pathlib import Path


class Memory:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,api_key_id INTEGER,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS procedures(id INTEGER PRIMARY KEY,tool_name TEXT,description TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY,namespace TEXT,text TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS api_keys(id INTEGER PRIMARY KEY,name TEXT,key_hash TEXT UNIQUE,prefix TEXT,active INTEGER DEFAULT 1,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS usage_events(id INTEGER PRIMARY KEY,api_key_id INTEGER,prompt_tokens INTEGER,completion_tokens INTEGER,total_tokens INTEGER,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS activity_log(id INTEGER PRIMARY KEY,api_key_id INTEGER,kind TEXT,detail TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS decision_shadow(id INTEGER PRIMARY KEY,api_key_id INTEGER,"
            "heuristic_route TEXT,cde_route TEXT,confidence REAL,abstained INTEGER,"
            "abstention_reason TEXT,suspected_ood INTEGER,normalized_entropy REAL,margin REAL,"
            "prompt_tokens INTEGER,completion_tokens INTEGER,total_tokens INTEGER,"
            "agreed INTEGER,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS oauth_states(state TEXT PRIMARY KEY,provider TEXT,gdpr_accepted INTEGER,"
            "cookie_consent TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS client_keys(id INTEGER PRIMARY KEY,owner_id INTEGER,name TEXT,prefix TEXT,"
            "key_hash TEXT UNIQUE,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,last_used_at DATETIME,revoked_at DATETIME)"
        )
        existing_ck = {row[1] for row in self.db.execute("PRAGMA table_info(client_keys)").fetchall()}
        for column in ("rpm", "tpm", "monthly_tokens"):
            if column not in existing_ck:
                self.db.execute(f"ALTER TABLE client_keys ADD COLUMN {column} INTEGER")
        self.db.execute("CREATE TABLE IF NOT EXISTS processed_stripe(id TEXT PRIMARY KEY, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self._migrate_usage_events()
        self._migrate_api_keys()
        self._migrate_messages()
        self._migrate_decision_shadow()
        self.db.commit()

    def _migrate_usage_events(self):
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(usage_events)").fetchall()}
        for column, decl in {"model": "TEXT", "latency_ms": "INTEGER", "status": "TEXT", "client_key_id": "INTEGER"}.items():
            if column not in existing:
                self.db.execute(f"ALTER TABLE usage_events ADD COLUMN {column} {decl}")

    def _migrate_decision_shadow(self):
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(decision_shadow)").fetchall()}
        additions = {
            "abstention_reason": "TEXT",
            "suspected_ood": "INTEGER",
            "normalized_entropy": "REAL",
            "margin": "REAL",
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "total_tokens": "INTEGER",
            "served_route": "TEXT",
            "routing_source": "TEXT",
            "fallback_reason": "TEXT",
            "rollout_bucket": "INTEGER",
            "cde_latency_ms": "INTEGER",
        }
        for column, decl in additions.items():
            if column not in existing:
                self.db.execute(f"ALTER TABLE decision_shadow ADD COLUMN {column} {decl}")

    def _migrate_api_keys(self):
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(api_keys)").fetchall()}
        additions = {
            "email": "TEXT",
            "trusted": "INTEGER DEFAULT 0",
            "role": "TEXT DEFAULT 'user'",
            "stripe_customer_id": "TEXT",
            "stripe_subscription_id": "TEXT",
            "token_balance": "INTEGER",
            "token_balance_max": "INTEGER",
            "password_hash": "TEXT",
            "oauth_provider": "TEXT",
            "oauth_subject": "TEXT",
            "gdpr_accepted_at": "DATETIME",
            "cookie_consent": "TEXT",
            "plan": "TEXT DEFAULT 'free'",
        }
        for column, decl in additions.items():
            if column not in existing:
                self.db.execute(f"ALTER TABLE api_keys ADD COLUMN {column} {decl}")
        if "role" not in existing:
            self.db.execute("UPDATE api_keys SET role='admin' WHERE trusted=1")

    def _migrate_messages(self):
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(messages)").fetchall()}
        if "api_key_id" not in existing:
            self.db.execute("ALTER TABLE messages ADD COLUMN api_key_id INTEGER")

    def create_api_key(self, name, key_hash, prefix, email=None, role="user",
                        stripe_customer_id=None, stripe_subscription_id=None):
        cur = self.db.execute(
            "INSERT INTO api_keys(name,key_hash,prefix,email,role,trusted,stripe_customer_id,stripe_subscription_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (name, key_hash, prefix, email, role, int(role == "admin"), stripe_customer_id, stripe_subscription_id),
        )
        self.db.commit()
        return cur.lastrowid

    def find_api_key(self, key_hash):
        row = self.db.execute(
            "SELECT id,name,active,role,stripe_customer_id,token_balance FROM api_keys WHERE key_hash=?",
            (key_hash,),
        ).fetchone()
        if not row:
            row = self.db.execute(
                "SELECT a.id,a.name,a.active,a.role,a.stripe_customer_id,a.token_balance,c.id,c.rpm,c.tpm,c.monthly_tokens FROM client_keys c "
                "JOIN api_keys a ON a.id=c.owner_id WHERE c.key_hash=? AND c.revoked_at IS NULL",
                (key_hash,),
            ).fetchone()
            if row:
                self.db.execute("UPDATE client_keys SET last_used_at=CURRENT_TIMESTAMP WHERE key_hash=?", (key_hash,))
                self.db.commit()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "active": bool(row[2]), "role": row[3] or "user",
                "stripe_customer_id": row[4], "token_balance": row[5],
                "client_key_id": row[6] if len(row) > 6 else None,
                "client_key_limits": ({"requests_per_minute": row[7], "tokens_per_minute": row[8],
                                       "monthly_tokens": row[9]} if len(row) > 6 else None)}

    def month_tokens_client(self, client_key_id):
        row = self.db.execute(
            "SELECT COALESCE(SUM(total_tokens),0) FROM usage_events WHERE client_key_id=? "
            "AND created_at>=date('now','start of month')", (client_key_id,)).fetchone()
        return row[0]

    def set_client_key_limits(self, owner_id, key_id, rpm, tpm, monthly):
        cur = self.db.execute(
            "UPDATE client_keys SET rpm=?,tpm=?,monthly_tokens=? WHERE id=? AND owner_id=? AND revoked_at IS NULL",
            (rpm, tpm, monthly, key_id, owner_id))
        self.db.commit()
        return cur.rowcount > 0

    def claim_stripe_event(self, event_id: str) -> bool:
        """True the first time an id (checkout session / invoice) is seen; False on replays, so credits are granted once."""
        cur = self.db.execute("INSERT OR IGNORE INTO processed_stripe(id) VALUES(?)", (event_id,))
        self.db.commit()
        return cur.rowcount > 0

    def get_plan(self, api_key_id):
        row = self.db.execute("SELECT plan FROM api_keys WHERE id=?", (api_key_id,)).fetchone()
        return row[0] if row else None

    def month_tokens(self, api_key_id):
        row = self.db.execute(
            "SELECT COALESCE(SUM(total_tokens),0) FROM usage_events WHERE api_key_id=? "
            "AND created_at>=date('now','start of month')", (api_key_id,)).fetchone()
        return row[0]

    def create_client_key(self, owner_id, name, key_hash, prefix, rpm=None, tpm=None, monthly_tokens=None):
        cur = self.db.execute(
            "INSERT INTO client_keys(owner_id,name,prefix,key_hash,rpm,tpm,monthly_tokens) VALUES(?,?,?,?,?,?,?)",
            (owner_id, name, prefix, key_hash, rpm, tpm, monthly_tokens))
        self.db.commit()
        return cur.lastrowid

    def list_client_keys(self, owner_id):
        rows = self.db.execute(
            "SELECT id,name,prefix,created_at,last_used_at,revoked_at,rpm,tpm,monthly_tokens FROM client_keys "
            "WHERE owner_id=? ORDER BY id DESC", (owner_id,)).fetchall()
        return [{"id": r[0], "name": r[1], "prefix": r[2], "created_at": r[3], "last_used_at": r[4],
                 "revoked": r[5] is not None,
                 "limits": {"requests_per_minute": r[6], "tokens_per_minute": r[7], "monthly_tokens": r[8]},
                 "month_tokens": self.month_tokens_client(r[0])} for r in rows]

    def revoke_client_key(self, owner_id, key_id):
        cur = self.db.execute(
            "UPDATE client_keys SET revoked_at=CURRENT_TIMESTAMP WHERE id=? AND owner_id=? AND revoked_at IS NULL",
            (key_id, owner_id))
        self.db.commit()
        return cur.rowcount > 0

    def usage_summary(self, api_key_id, days=30):
        since = f"-{int(days)} days"
        total = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),"
            "COALESCE(SUM(total_tokens),0),AVG(latency_ms) FROM usage_events "
            "WHERE api_key_id=? AND created_at>=datetime('now',?)", (api_key_id, since)).fetchone()
        daily = self.db.execute(
            "SELECT date(created_at),COUNT(*),COALESCE(SUM(total_tokens),0) FROM usage_events "
            "WHERE api_key_id=? AND created_at>=datetime('now',?) GROUP BY 1 ORDER BY 1", (api_key_id, since)).fetchall()
        by_key = self.db.execute(
            "SELECT u.client_key_id,c.name,c.prefix,COUNT(*),COALESCE(SUM(u.total_tokens),0) FROM usage_events u "
            "LEFT JOIN client_keys c ON c.id=u.client_key_id WHERE u.api_key_id=? AND u.created_at>=datetime('now',?) "
            "GROUP BY u.client_key_id ORDER BY 5 DESC", (api_key_id, since)).fetchall()
        return {"by_key": [{"client_key_id": kid, "name": name or "Clave principal (login)", "prefix": prefix,
                            "requests": req, "tokens": tok} for kid, name, prefix, req, tok in by_key],
                "days": days, "requests": total[0], "prompt_tokens": total[1], "completion_tokens": total[2],
                "total_tokens": total[3], "avg_latency_ms": int(total[4]) if total[4] is not None else None,
                "daily": [{"date": d, "requests": r, "tokens": t} for d, r, t in daily]}

    def get_api_key(self, key_id):
        row = self.db.execute(
            "SELECT id,name,prefix,active,role,email,stripe_customer_id,stripe_subscription_id,created_at,"
            "token_balance,token_balance_max,plan FROM api_keys WHERE id=?",
            (key_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "name": row[1], "prefix": row[2], "active": bool(row[3]), "role": row[4] or "user",
            "email": row[5], "stripe_customer_id": row[6], "stripe_subscription_id": row[7], "created_at": row[8],
            "token_balance": row[9], "token_balance_max": row[10], "plan": row[11] or "free",
        }

    def set_plan(self, key_id, plan):
        self.db.execute("UPDATE api_keys SET plan=? WHERE id=?", (plan, key_id))
        self.db.commit()

    def find_by_email(self, email):
        row = self.db.execute("SELECT id FROM api_keys WHERE email=?", (email,)).fetchone()
        return row[0] if row else None

    def find_by_oauth(self, provider, subject):
        row = self.db.execute(
            "SELECT id FROM api_keys WHERE oauth_provider=? AND oauth_subject=?", (provider, subject)
        ).fetchone()
        return row[0] if row else None

    def create_account(self, name, email, key_hash, prefix, role="user", password_hash=None,
                        oauth_provider=None, oauth_subject=None, gdpr_accepted=False, cookie_consent=None,
                        plan="free"):
        cur = self.db.execute(
            "INSERT INTO api_keys(name,key_hash,prefix,email,role,trusted,password_hash,oauth_provider,"
            "oauth_subject,cookie_consent,plan) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (name, key_hash, prefix, email, role, int(role == "admin"), password_hash, oauth_provider,
             oauth_subject, json.dumps(cookie_consent) if cookie_consent else None, plan),
        )
        if gdpr_accepted:
            self.db.execute(
                "UPDATE api_keys SET gdpr_accepted_at=CURRENT_TIMESTAMP WHERE id=?", (cur.lastrowid,)
            )
        self.db.commit()
        return cur.lastrowid

    def get_account_credentials(self, email):
        row = self.db.execute(
            "SELECT id,password_hash,active FROM api_keys WHERE email=?", (email,)
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "password_hash": row[1], "active": bool(row[2])}

    def get_password_hash(self, key_id):
        row = self.db.execute("SELECT password_hash FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return row[0] if row else None

    def set_password_hash(self, key_id, password_hash):
        self.db.execute("UPDATE api_keys SET password_hash=? WHERE id=?", (password_hash, key_id))
        self.db.commit()

    def rotate_api_key(self, key_id, key_hash, prefix):
        self.db.execute("UPDATE api_keys SET key_hash=?, prefix=?, active=1 WHERE id=?", (key_hash, prefix, key_id))
        self.db.commit()

    def record_consent(self, key_id, gdpr_accepted, cookie_consent):
        self.db.execute(
            "UPDATE api_keys SET gdpr_accepted_at=CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE gdpr_accepted_at END, "
            "cookie_consent=? WHERE id=?",
            (int(bool(gdpr_accepted)), json.dumps(cookie_consent) if cookie_consent else None, key_id),
        )
        self.db.commit()

    def save_oauth_state(self, state, provider, gdpr_accepted, cookie_consent):
        self.db.execute(
            "INSERT INTO oauth_states(state,provider,gdpr_accepted,cookie_consent) VALUES(?,?,?,?)",
            (state, provider, int(bool(gdpr_accepted)), json.dumps(cookie_consent) if cookie_consent else None),
        )
        self.db.commit()

    def pop_oauth_state(self, state):
        row = self.db.execute(
            "SELECT provider,gdpr_accepted,cookie_consent FROM oauth_states WHERE state=?", (state,)
        ).fetchone()
        if not row:
            return None
        self.db.execute("DELETE FROM oauth_states WHERE state=?", (state,))
        self.db.commit()
        return {"provider": row[0], "gdpr_accepted": bool(row[1]), "cookie_consent": json.loads(row[2]) if row[2] else None}

    def link_stripe_customer(self, key_id, stripe_customer_id, stripe_subscription_id):
        self.db.execute(
            "UPDATE api_keys SET stripe_customer_id=?, stripe_subscription_id=? WHERE id=?",
            (stripe_customer_id, stripe_subscription_id, key_id),
        )
        self.db.commit()

    def find_api_key_by_customer(self, stripe_customer_id):
        row = self.db.execute(
            "SELECT id FROM api_keys WHERE stripe_customer_id=?",
            (stripe_customer_id,),
        ).fetchone()
        return row[0] if row else None

    def list_api_keys(self):
        rows = self.db.execute(
            "SELECT id,name,prefix,active,role,email,created_at FROM api_keys ORDER BY id DESC"
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "prefix": r[2], "active": bool(r[3]), "role": r[4] or "user",
             "email": r[5], "created_at": r[6]}
            for r in rows
        ]

    def revoke_api_key(self, key_id):
        cur = self.db.execute("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))
        self.db.commit()
        return cur.rowcount > 0

    def set_api_key_active(self, key_id, active):
        cur = self.db.execute("UPDATE api_keys SET active=? WHERE id=?", (int(active), key_id))
        self.db.commit()
        return cur.rowcount > 0

    def set_api_key_role(self, key_id, role):
        cur = self.db.execute("UPDATE api_keys SET role=?, trusted=? WHERE id=?", (role, int(role == "admin"), key_id))
        self.db.commit()
        return cur.rowcount > 0

    def add_token_credit(self, key_id, amount):
        cur = self.db.execute(
            "UPDATE api_keys SET "
            "token_balance = COALESCE(token_balance, 0) + ?, "
            "token_balance_max = COALESCE(token_balance_max, 0) + ? "
            "WHERE id=?",
            (amount, amount, key_id),
        )
        self.db.commit()
        return cur.rowcount > 0

    def decrement_token_balance(self, key_id, amount):
        self.db.execute(
            "UPDATE api_keys SET token_balance = MAX(token_balance - ?, 0) "
            "WHERE id=? AND token_balance IS NOT NULL",
            (amount, key_id),
        )
        self.db.commit()

    def record_usage(self, api_key_id, prompt_tokens, completion_tokens, model=None, latency_ms=None, status="ok",
                     client_key_id=None):
        self.db.execute(
            "INSERT INTO usage_events(api_key_id,prompt_tokens,completion_tokens,total_tokens,model,latency_ms,status,client_key_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (api_key_id, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, model, latency_ms, status,
             client_key_id),
        )
        self.db.commit()

    def record_decision_shadow(self, api_key_id, heuristic_route, cde_route, confidence, abstained,
                               abstention_reason=None, suspected_ood=None, normalized_entropy=None, margin=None,
                               prompt_tokens=None, completion_tokens=None, total_tokens=None,
                               served_route=None, routing_source=None, fallback_reason=None,
                               rollout_bucket=None, cde_latency_ms=None):
        agreed = bool(cde_route and cde_route == heuristic_route and not abstained)
        self.db.execute(
            "INSERT INTO decision_shadow(api_key_id,heuristic_route,cde_route,confidence,abstained,"
            "abstention_reason,suspected_ood,normalized_entropy,margin,prompt_tokens,completion_tokens,total_tokens,"
            "served_route,routing_source,fallback_reason,rollout_bucket,cde_latency_ms,agreed) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                api_key_id,
                heuristic_route,
                cde_route,
                float(confidence),
                int(bool(abstained)),
                abstention_reason,
                int(bool(suspected_ood)) if suspected_ood is not None else None,
                float(normalized_entropy) if normalized_entropy is not None else None,
                float(margin) if margin is not None else None,
                int(prompt_tokens) if prompt_tokens is not None else None,
                int(completion_tokens) if completion_tokens is not None else None,
                int(total_tokens) if total_tokens is not None else None,
                served_route,
                routing_source,
                fallback_reason,
                int(rollout_bucket) if rollout_bucket is not None else None,
                int(cde_latency_ms) if cde_latency_ms is not None else None,
                int(agreed),
            ),
        )
        self.db.commit()

    def decision_shadow_summary(self, days=30):
        since = f"-{max(1, int(days))} days"
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(agreed),0),COALESCE(SUM(abstained),0),"
            "COALESCE(SUM(suspected_ood),0),AVG(confidence),AVG(normalized_entropy),AVG(margin),"
            "COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),COALESCE(SUM(total_tokens),0) "
            "FROM decision_shadow WHERE created_at>=datetime('now',?)",
            (since,),
        ).fetchone()
        (
            total, agreed, abstained, suspected_ood, avg_confidence, avg_entropy, avg_margin,
            prompt_tokens, completion_tokens, total_tokens,
        ) = row
        disagreements = self.db.execute(
            "SELECT heuristic_route,cde_route,COUNT(*) FROM decision_shadow "
            "WHERE created_at>=datetime('now',?) AND abstained=0 AND heuristic_route<>cde_route "
            "GROUP BY heuristic_route,cde_route ORDER BY COUNT(*) DESC LIMIT 20", (since,)
        ).fetchall()
        abstention_reasons = self.db.execute(
            "SELECT COALESCE(abstention_reason,'unknown'),COUNT(*) FROM decision_shadow "
            "WHERE created_at>=datetime('now',?) AND abstained=1 GROUP BY abstention_reason ORDER BY COUNT(*) DESC", (since,)
        ).fetchall()
        routing_sources = self.db.execute(
            "SELECT COALESCE(routing_source,'legacy'),COUNT(*) FROM decision_shadow "
            "WHERE created_at>=datetime('now',?) GROUP BY routing_source ORDER BY COUNT(*) DESC", (since,)
        ).fetchall()
        fallback_reasons = self.db.execute(
            "SELECT COALESCE(fallback_reason,'unknown'),COUNT(*) FROM decision_shadow "
            "WHERE created_at>=datetime('now',?) AND fallback_reason IS NOT NULL "
            "GROUP BY fallback_reason ORDER BY COUNT(*) DESC", (since,)
        ).fetchall()
        latency_row = self.db.execute(
            "SELECT AVG(cde_latency_ms) FROM decision_shadow "
            "WHERE created_at>=datetime('now',?) AND cde_latency_ms IS NOT NULL", (since,)
        ).fetchone()
        return {
            "days": max(1, int(days)), "samples": total, "agreements": agreed,
            "agreement_rate": (agreed / total) if total else None,
            "abstentions": abstained, "abstention_rate": (abstained / total) if total else None,
            "suspected_ood": suspected_ood,
            "suspected_ood_rate": (suspected_ood / total) if total else None,
            "avg_confidence": avg_confidence, "avg_normalized_entropy": avg_entropy, "avg_margin": avg_margin,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": (total_tokens / total) if total else None,
            "abstention_reasons": {reason: count for reason, count in abstention_reasons},
            "routing_sources": {source: count for source, count in routing_sources},
            "fallback_reasons": {reason: count for reason, count in fallback_reasons},
            "avg_cde_latency_ms": latency_row[0] if latency_row else None,
            "top_disagreements": [{"heuristic": a, "cde": b, "count": n} for a,b,n in disagreements],
        }

    def record_activity(self, api_key_id, kind, detail):
        self.db.execute(
            "INSERT INTO activity_log(api_key_id,kind,detail) VALUES(?,?,?)",
            (api_key_id, kind, detail),
        )
        self.db.commit()

    def list_activity(self, api_key_id=None, kind=None, limit=100):
        clauses, params = [], []
        if api_key_id is not None:
            clauses.append("api_key_id=?")
            params.append(api_key_id)
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        sql = "SELECT id,api_key_id,kind,detail,created_at FROM activity_log"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(sql, tuple(params)).fetchall()
        return [
            {"id": r[0], "api_key_id": r[1], "kind": r[2], "detail": r[3], "created_at": r[4]}
            for r in rows
        ]

    def sum_usage(self, api_key_id):
        row = self.db.execute(
            "SELECT COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),COALESCE(SUM(total_tokens),0),COUNT(*) "
            "FROM usage_events WHERE api_key_id=?",
            (api_key_id,),
        ).fetchone()
        return {"prompt_tokens": row[0], "completion_tokens": row[1], "total_tokens": row[2], "requests": row[3]}

    def add(self, sid, role, content, api_key_id=None):
        self.db.execute(
            "INSERT INTO messages(session_id,role,content,api_key_id) VALUES(?,?,?,?)",
            (sid, role, content, api_key_id),
        )
        self.db.commit()

    def history(self, sid, limit=16, api_key_id=None):
        """Only returns messages owned by `api_key_id`, so a client-chosen session_id can't read another user's chat."""
        rows = self.db.execute(
            "SELECT role,content FROM messages WHERE session_id=? AND api_key_id IS ? ORDER BY id DESC LIMIT ?",
            (sid, api_key_id, limit),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def list_conversations(self, api_key_id, limit=50):
        rows = self.db.execute(
            "SELECT session_id, MAX(created_at) as last_at, COUNT(*) as n, "
            "(SELECT content FROM messages m2 WHERE m2.session_id=m1.session_id ORDER BY m2.id DESC LIMIT 1) as last_message "
            "FROM messages m1 WHERE api_key_id=? GROUP BY session_id ORDER BY last_at DESC LIMIT ?",
            (api_key_id, limit),
        ).fetchall()
        return [
            {"session_id": r[0], "last_at": r[1], "message_count": r[2], "last_message": (r[3] or "")[:200]}
            for r in rows
        ]

    def delete_conversation(self, api_key_id, session_id):
        cur = self.db.execute(
            "DELETE FROM messages WHERE api_key_id=? AND session_id=?", (api_key_id, session_id)
        )
        self.db.commit()
        return cur.rowcount > 0

    def prune_conversations(self, api_key_id, keep=10):
        rows = self.db.execute(
            "SELECT session_id, MAX(id) as last_id FROM messages WHERE api_key_id=? "
            "GROUP BY session_id ORDER BY last_id DESC LIMIT -1 OFFSET ?",
            (api_key_id, keep),
        ).fetchall()
        for (session_id, _) in rows:
            self.db.execute(
                "DELETE FROM messages WHERE api_key_id=? AND session_id=?", (api_key_id, session_id)
            )
        if rows:
            self.db.commit()

    def get_conversation(self, api_key_id, session_id, limit=200):
        rows = self.db.execute(
            "SELECT role,content,created_at FROM messages WHERE session_id=? AND api_key_id=? ORDER BY id ASC LIMIT ?",
            (session_id, api_key_id, limit),
        ).fetchall()
        return [{"role": r, "content": c, "created_at": t} for r, c, t in rows]

    def search(self, sid, query, limit=5):
        q = f"%{query.lower()}%"
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id=? AND LOWER(content) LIKE ? ORDER BY id DESC LIMIT ?",
            (sid, q, limit),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def record_procedure(self, tool_name, description):
        self.db.execute(
            "INSERT INTO procedures(tool_name,description) VALUES(?,?)",
            (tool_name, description),
        )
        self.db.commit()

    def get_procedures(self, tool_name, limit=10):
        rows = self.db.execute(
            "SELECT tool_name, description FROM procedures WHERE tool_name=? ORDER BY id DESC LIMIT ?",
            (tool_name, limit),
        ).fetchall()
        return [{"tool_name": tool, "description": desc} for tool, desc in rows]

    def add_document(self, namespace, text):
        if not text or not text.strip():
            raise ValueError("document text cannot be empty")
        self.db.execute(
            "INSERT INTO documents(namespace,text) VALUES(?,?)",
            (namespace, text.strip()),
        )
        self.db.commit()

    def export_user_data(self, api_key_id):
        """RGPD art. 20 (portabilidad): todos los datos personales ligados a una clave."""
        account = self.get_api_key(api_key_id)
        messages = self.db.execute(
            "SELECT session_id,role,content,created_at FROM messages WHERE api_key_id=? ORDER BY id ASC",
            (api_key_id,),
        ).fetchall()
        usage = self.db.execute(
            "SELECT prompt_tokens,completion_tokens,total_tokens,created_at FROM usage_events WHERE api_key_id=? ORDER BY id ASC",
            (api_key_id,),
        ).fetchall()
        activity = self.db.execute(
            "SELECT kind,detail,created_at FROM activity_log WHERE api_key_id=? ORDER BY id ASC",
            (api_key_id,),
        ).fetchall()
        return {
            "account": account,
            "messages": [{"session_id": s, "role": r, "content": c, "created_at": t} for s, r, c, t in messages],
            "usage_events": [{"prompt_tokens": p, "completion_tokens": c, "total_tokens": t, "created_at": ts} for p, c, t, ts in usage],
            "activity_log": [{"kind": k, "detail": d, "created_at": t} for k, d, t in activity],
        }

    def delete_user_data(self, api_key_id):
        """RGPD art. 17 (derecho al olvido): borra historial/actividad y anonimiza la cuenta."""
        self.db.execute("DELETE FROM messages WHERE api_key_id=?", (api_key_id,))
        self.db.execute("DELETE FROM usage_events WHERE api_key_id=?", (api_key_id,))
        self.db.execute("DELETE FROM activity_log WHERE api_key_id=?", (api_key_id,))
        self.db.execute(
            "UPDATE api_keys SET name='[eliminado]', email=NULL, active=0 WHERE id=?",
            (api_key_id,),
        )
        self.db.commit()

    def purge_older_than(self, days):
        """Política de retención: elimina mensajes y logs de actividad más antiguos que `days`."""
        cur_messages = self.db.execute(
            "DELETE FROM messages WHERE created_at < datetime('now', ?)", (f"-{int(days)} days",)
        )
        cur_activity = self.db.execute(
            "DELETE FROM activity_log WHERE created_at < datetime('now', ?)", (f"-{int(days)} days",)
        )
        self.db.commit()
        return {"messages_deleted": cur_messages.rowcount, "activity_deleted": cur_activity.rowcount}

    def semantic_search(self, query, namespace=None, limit=5):
        synonyms = {
            "math": ["math", "arithmetic", "calculation", "calculator"],
            "verify": ["verify", "verification", "validate", "validates", "check", "confirm"],
            "tool": ["tool", "tools", "assistant", "calculator", "execution"],
            "use": ["use", "using", "usage", "tool", "calculator"],
        }
        extracted = [w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(w) > 2]
        words = []
        for word in extracted:
            words.append(word)
            words.extend(synonyms.get(word, []))
        words = list(dict.fromkeys(words))
        if not words:
            return []

        all_clauses = []
        params = []
        for word in words:
            all_clauses.append("LOWER(text) LIKE ?")
            params.append(f"%{word}%")
        if namespace:
            all_clauses.append("namespace = ?")
            params.append(namespace)
        sql = "SELECT text FROM documents"
        if all_clauses:
            sql += " WHERE (" + " OR ".join(all_clauses) + ")"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.execute(sql, tuple(params)).fetchall()
        return [{"text": text} for (text,) in rows]

