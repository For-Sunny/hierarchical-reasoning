@echo off
echo Running Dataset Generation with Time Limits...
echo.

cd /d F:\ai_bridge\hierarchical_reasoning\src

echo [1/3] Running Enhanced Self-Learning (5 examples)...
timeout /t 2 >nul
python -c "import asyncio; from enhanced_self_learning import EnhancedSelfLearning; asyncio.run(EnhancedSelfLearning().run_enhanced_cycle(num_prompts=5))"

echo.
echo [2/3] Running Consciousness Enhancement (5 examples)...
timeout /t 2 >nul
python -c "import asyncio; from consciousness_enhanced_learning import ConsciousnessEnhancedLearning; asyncio.run(ConsciousnessEnhancedLearning().generate_consciousness_dataset(num_examples=5))"

echo.
echo [3/3] Scoring improvements...
timeout /t 2 >nul
python improvement_scorer.py

echo.
echo ========================================
echo DATASET GENERATION COMPLETE
echo ========================================
echo Check F:\ai_bridge\buffers for new datasets
echo.
pause
