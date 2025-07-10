export KERAS_BACKEND="tensorflow"


alias syncdb="uv run python -m src.zip_mnist.schema"
alias xgb="uv run python -m src.zip_mnist.xgb"
alias runall="uv run python -m src.zip_mnist.run_all"
alias cleardb="uv run python -m src.zip_mnist.clear_db"
alias showdb="uv run show_sqlite.py"
alias rmdb="uv run python -m src.zip_mnist.rmdb"
alias auto="uv run python -m src.zip_mnist.auto"