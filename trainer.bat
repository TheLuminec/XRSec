@echo off
set DATA_DIR="datasets\VR_User_Behavior_Dataset_(Spherical_Video_Streaming)"
set EPOCHS=20
set BATCH_SIZE=8192
set LR=0.001
set SEED=42

python "model\main.py" ^
--data-dir "%DATA_DIR%\processed_data\users" ^
--epochs %EPOCHS% ^
--batch-size %BATCH_SIZE% ^
--lr %LR% ^
--seed %SEED% ^
--mode "train" ^
--save-path "%DATA_DIR%\processed_data\trained_model_v2.pth" ^
--graph ^
--graph-path "%DATA_DIR%\processed_data\training_history_v2.png"