@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

rem Launch two agent API servers in separate windows
start "Agent A (8081)" cmd /k python -c "import asyncio; from phase4_api_server import Phase4APIServer; config={'agent_id':'agent_A','db_path':'agent_A.db','human_feedback_enabled':True,'peers':['http://127.0.0.1:8082','http://127.0.0.1:8083'],'heartbeat_interval_s':5,'leader_id':'agent_A'}; server=Phase4APIServer(config, port=8081); asyncio.run(server.start_server())"

start "Agent B (8082)" cmd /k python -c "import asyncio; from phase4_api_server import Phase4APIServer; config={'agent_id':'agent_B','db_path':'agent_B.db','human_feedback_enabled':True,'peers':['http://127.0.0.1:8081','http://127.0.0.1:8083'],'heartbeat_interval_s':5,'leader_id':'agent_A','leader_url':'http://127.0.0.1:8081'}; server=Phase4APIServer(config, port=8082); asyncio.run(server.start_server())"

start "Agent C (8083)" cmd /k python -c "import asyncio; from phase4_api_server import Phase4APIServer; config={'agent_id':'agent_C','db_path':'agent_C.db','human_feedback_enabled':True,'peers':['http://127.0.0.1:8081','http://127.0.0.1:8082'],'heartbeat_interval_s':5,'leader_id':'agent_A','leader_url':'http://127.0.0.1:8081'}; server=Phase4APIServer(config, port=8083); asyncio.run(server.start_server())"

echo Started Agent A on 8081, Agent B on 8082, and Agent C on 8083 in separate windows.
echo Training is now manual. Use the CLI to start/pause/resume/stop.
echo Example: python cli_control.py start --agent ALL
echo Close those windows to stop the servers.
exit /b 0


