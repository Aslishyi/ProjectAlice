@echo off
chcp 65001 >nul
echo Starting NapCat and AliceBot...

call "D:/anaconda3/Scripts/activate.bat"
call conda activate py311
start "NapCat" cmd /k "cd /d e:\Local_GitHub\DevCode\ProjectAlice\NapCat.41785.Shell && call napcat.bat"

REM Start AliceBot in a separate terminal window
start "AliceBot" cmd /k "cd /d e:\Local_GitHub\DevCode\ProjectAlice\AliceBot && python qq_server.py"

echo NapCat and AliceBot have been started. Please check the two terminal windows.
