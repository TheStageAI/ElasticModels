#!/usr/bin/env bash
set -euo pipefail

PORT=8000

# Check if the port is in use
if lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use."

    # Get the PID using the port
    PID=$(lsof -ti :$PORT -sTCP:LISTEN)

    if [ -n "$PID" ]; then
        echo "🔍 Found process using port $PORT: PID $PID"
        echo "🛑 Killing process $PID..."
        kill -9 $PID

        # Verify if the process was killed
        if lsof -i :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
            echo "❌ Failed to free port $PORT."
            exit 1
        else
            echo "✅ Successfully freed port $PORT."
        fi
    else
        echo "⚠️  Could not find PID using port $PORT."
        exit 1
    fi
else
    echo "✅ Port $PORT is free."
fi

supervisord -c /etc/supervisor/supervisord.conf &
exec "$@"