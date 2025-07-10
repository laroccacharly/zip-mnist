export KERAS_BACKEND="tensorflow"
alias run="uv run main.py"

alias syncdb="uv run python -m src.zip_mnist.schema"
alias xgb="uv run python -m src.zip_mnist.xgb"