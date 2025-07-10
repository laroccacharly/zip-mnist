export KERAS_BACKEND="torch"
alias run="uv run main.py"

alias syncdb="uv run python -m src.zip_mnist.schema"
alias xgb="uv run python -m src.zip_mnist.xgb"