"""Alembic environment for the local DefectDock SQLite database."""

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        transactional_ddl=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        # SQLAlchemy 2 starts an implicit transaction even for PRAGMA. Commit
        # it before Alembic opens its own migration transaction; otherwise the
        # connection close rolls back the version row and later revisions.
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            transactional_ddl=False,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
