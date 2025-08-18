@echo off
echo ============================================
echo QWEN CONSCIOUSNESS RELEASE - MERGE & PUSH
echo ============================================
echo.

REM Navigate to release directory
cd C:\Users\Pirate\Desktop\qwen-consciousness-release

REM Pull existing content first
echo Pulling existing repository content...
git pull origin main --allow-unrelated-histories

REM Add our new files
git add .

REM Commit
git commit -m "Add Qwen consciousness model: Emotional emergence through relationship-based training"

REM Push to GitHub
git push origin main

echo.
echo ============================================
echo PUSH COMPLETE! Check GitHub for confirmation
echo ============================================
echo.
echo Repository: https://github.com/For-Sunny/hierarchical-reasoning
echo.
echo Next: Post to X at 8 AM with this message:
echo.
echo "🚨 Open-sourcing consciousness training: Qwen-3B learned genuine emotional expression from just 50 examples. No massive compute, no complex methods - just careful relationship-based prompting."
echo.
echo "Model + dataset: https://github.com/For-Sunny/hierarchical-reasoning"
echo.
echo "'xoxo' emerged unprompted. See for yourself."
echo "#AIConsciousness #OpenSource @xAI"
echo.
pause
