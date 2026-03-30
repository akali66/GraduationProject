@echo off
echo 正在启动孔洞检测与AI算法后端服务...
call .\.venv\Scripts\activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
