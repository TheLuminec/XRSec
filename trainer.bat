@echo off
set DATA_DIR=datasets\VR_User_Behavior_Dataset_(Spherical_Video_Streaming)
set EPOCHS=100
set BATCH_SIZE=8192
set LR=0.001
set SEED=67
set EMBEDDING_DIM=128

.venv\Scripts\python.exe "model\main.py" "data_dirs=['%DATA_DIR%\processed_data\users']" epochs=%EPOCHS% batch_size=%BATCH_SIZE% lr=%LR% seed=%SEED% embedding_dim=%EMBEDDING_DIM% mode=train "save_path='%DATA_DIR%\processed_data\trained_model_v2.pth'" graph=true "graph_path='%DATA_DIR%\processed_data\training_history_v2.png'"