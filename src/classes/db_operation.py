from common.const import (CVS_DB_HOST, CVS_DB_NAME, CVS_DB_PASSWORD,
                          CVS_DB_PORT, CVS_DB_USER)
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import text


class DBOperation:
    def __init__(self, host, port, username, password, database) -> None:
        self.engine = create_engine(
            f"postgresql://{username}:{password}@{host}:{port}/{database}"
        )

    def get_session(self,) -> Session:
        return sessionmaker(bind=self.engine)()

    def execute_statement(self, method: str, statement):
        method = method.lower()
        if method not in ["sql", "statement"]:
            raise ValueError("method must be 'sql' or 'statement'")
        session = self.get_session()
        return session.execute(text(statement)) if method == 'sql' \
            else session.execute(statement)

    def do_upsert(
        self,
        table_object,
        records: list[dict],
        index_elements: list,
        update_columns: list
    ) -> None:
        stmt = postgresql.insert(table_object).values(records)
        set_ = {
            column: getattr(stmt.excluded, column) for column in update_columns
            }
        stmt = stmt.on_conflict_do_update(
            index_elements=index_elements,
            set_=set_
        )
        session = self.get_session()
        session.execute(stmt)
        session.commit()

    def do_insert(
        self,
        table_object,
        records: list[dict],
        index_elements: list
    ) -> None:
        stmt = postgresql.insert(table_object).values(records) \
            .on_conflict_do_nothing(index_elements=index_elements)
        session = self.get_session()
        session.execute(stmt)
        session.commit()


db_operator = DBOperation(
    CVS_DB_HOST, CVS_DB_PORT,
    CVS_DB_USER, CVS_DB_PASSWORD, CVS_DB_NAME
)
