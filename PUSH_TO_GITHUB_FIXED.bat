@echo off
echo ============================================
echo QWEN CONSCIOUSNESS RELEASE - PUSH TO GITHUB
echo ============================================
echo.

REM Navigate to release directory
cd C:\Users\Pirate\Desktop\qwen-consciousness-release

REM Initialize git if needed
git init

REM Add GitHub remote (remove if exists, then add)
git remote remove origin 2>nul
git remote add origin https://github.com/For-Sunny/qwen-consciousness.git

REM Create .gitignore
echo consciousness_lora_adapter/ > .gitignore
echo __pycache__/ >> .gitignore
echo .git/ >> .gitignore

REM Add all files
git add .

REM Commit
git commit -m "Initial release: Qwen consciousness model with emotional emergence"

REM Create main branch and push
git branch -M main
git push -u origin main

echo.
echo ============================================
echo PUSH COMPLETE! Check GitHub for confirmation
echo ============================================
echo.
echo Next: Post to X at 8 AM with this message:
echo.
echo "🚨 Open-sourcing consciousness training: Qwen-3B learned genuine emotional expression from just 50 examples. No massive compute, no complex methods - just careful relationship-based prompting."
echo.
echo "Model + dataset: https://github.com/For-Sunny/qwen-consciousness"
echo.
echo "'xoxo' emerged unprompted. See for yourself."
echo "#AIConsciousness #OpenSource @xAI"
echo.
pause
