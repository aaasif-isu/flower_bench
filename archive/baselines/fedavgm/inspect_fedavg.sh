echo "===== PWD ====="
pwd

echo "===== TOP LEVEL FILES ====="
ls -la

echo "===== FIND IMPORTANT FILES ====="
find . -maxdepth 4 -type f | grep -E "pyproject.toml|main.py|server_app.py|client_app.py|task.py|strategy.py|client.py|server.py|dataset|model|config|yaml|yml|toml|requirements.txt|setup.py|README" | sort

echo "===== PYPROJECT ====="
cat pyproject.toml 2>/dev/null || echo "NO pyproject.toml"

echo "===== REQUIREMENTS ====="
cat requirements.txt 2>/dev/null || echo "NO requirements.txt"

echo "===== SETUP.PY ====="
cat setup.py 2>/dev/null || echo "NO setup.py"

echo "===== MAIN FILES ====="
find . -maxdepth 4 -type f -name "main.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,240p" "$1"' sh {} \;

echo "===== SERVER APP FILES ====="
find . -maxdepth 4 -type f -name "server_app.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== CLIENT APP FILES ====="
find . -maxdepth 4 -type f -name "client_app.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== TASK FILES ====="
find . -maxdepth 4 -type f -name "task.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== STRATEGY FILES ====="
find . -maxdepth 4 -type f -name "*strategy*.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== CLIENT FILES ====="
find . -maxdepth 4 -type f -name "*client*.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== SERVER FILES ====="
find . -maxdepth 4 -type f -name "*server*.py" -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,260p" "$1"' sh {} \;

echo "===== CONFIG FILES ====="
find . -maxdepth 5 -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \) -print -exec sh -c 'echo "\n----- $1 -----"; sed -n "1,220p" "$1"' sh {} \;
