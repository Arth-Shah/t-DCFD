{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d45270f6",
   "metadata": {
    "papermill": {
     "duration": 0.003906,
     "end_time": "2026-05-25T07:57:48.236226+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:48.232320+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# 1. Imports + Config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "799e2f25",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:48.243528Z",
     "iopub.status.busy": "2026-05-25T07:57:48.243219Z",
     "iopub.status.idle": "2026-05-25T07:57:52.717804Z",
     "shell.execute_reply": "2026-05-25T07:57:52.716833Z"
    },
    "papermill": {
     "duration": 4.480229,
     "end_time": "2026-05-25T07:57:52.719678+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:48.239449+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import librosa\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader, random_split\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "237cb3d7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:52.727262Z",
     "iopub.status.busy": "2026-05-25T07:57:52.726958Z",
     "iopub.status.idle": "2026-05-25T07:57:52.731309Z",
     "shell.execute_reply": "2026-05-25T07:57:52.730533Z"
    },
    "papermill": {
     "duration": 0.009865,
     "end_time": "2026-05-25T07:57:52.732840+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:52.722975+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ROOT_PATH = \"/kaggle/input/datasets/artharking/severity-dau-kdah/Severity_DAU_KDAH\"  # change this\n",
    "\n",
    "SR = 16000\n",
    "N_MFCC = 20\n",
    "FRAME_LEN = 0.025  # 25 ms\n",
    "HOP_LEN = 0.010    # 10 ms\n",
    "\n",
    "FIXED_LEN = 398  # time frames (important for CNN)\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 20\n",
    "LR = 1e-3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4564399e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:52.740037Z",
     "iopub.status.busy": "2026-05-25T07:57:52.739709Z",
     "iopub.status.idle": "2026-05-25T07:57:52.743205Z",
     "shell.execute_reply": "2026-05-25T07:57:52.742636Z"
    },
    "papermill": {
     "duration": 0.008642,
     "end_time": "2026-05-25T07:57:52.744434+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:52.735792+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "label_map = {\n",
    "    \"N\": 0,\n",
    "    \"L\": 1,\n",
    "    \"M\": 2,\n",
    "    \"H\": 3\n",
    "}\n",
    "NUM_CLASSES = len(label_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b664e429",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:52.752920Z",
     "iopub.status.busy": "2026-05-25T07:57:52.752633Z",
     "iopub.status.idle": "2026-05-25T07:57:52.758362Z",
     "shell.execute_reply": "2026-05-25T07:57:52.757835Z"
    },
    "papermill": {
     "duration": 0.010932,
     "end_time": "2026-05-25T07:57:52.759627+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:52.748695+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import librosa\n",
    "from tqdm import tqdm\n",
    "\n",
    "SAVE_PATH = \"/kaggle/working/mfcc_features\"\n",
    "os.makedirs(SAVE_PATH, exist_ok=True)\n",
    "\n",
    "def save_mfcc_dataset(root_dir, split):\n",
    "    split_path = os.path.join(root_dir, split)\n",
    "\n",
    "    for severity in os.listdir(split_path):\n",
    "        sev_path = os.path.join(split_path, severity)\n",
    "\n",
    "        if not os.path.isdir(sev_path):\n",
    "            continue\n",
    "\n",
    "        save_sev_path = os.path.join(SAVE_PATH, split, severity)\n",
    "        os.makedirs(save_sev_path, exist_ok=True)\n",
    "\n",
    "        for file in tqdm(os.listdir(sev_path), desc=f\"{split}-{severity}\"):\n",
    "            if not file.endswith(\".wav\"):\n",
    "                continue\n",
    "\n",
    "            file_path = os.path.join(sev_path, file)\n",
    "\n",
    "            mfcc = extract_mfcc(file_path)\n",
    "\n",
    "            save_file = os.path.join(save_sev_path, file.replace(\".wav\", \".npy\"))\n",
    "            np.save(save_file, mfcc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bc86ae7d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:52.766976Z",
     "iopub.status.busy": "2026-05-25T07:57:52.766282Z",
     "iopub.status.idle": "2026-05-25T07:57:52.771114Z",
     "shell.execute_reply": "2026-05-25T07:57:52.770548Z"
    },
    "papermill": {
     "duration": 0.009885,
     "end_time": "2026-05-25T07:57:52.772414+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:52.762529+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def extract_mfcc(file_path):\n",
    "    y, sr = librosa.load(file_path, sr=SR)\n",
    "\n",
    "    mfcc = librosa.feature.mfcc(\n",
    "        y=y,\n",
    "        sr=sr,\n",
    "        n_mfcc=N_MFCC,\n",
    "        n_fft=int(FRAME_LEN * sr),\n",
    "        hop_length=int(HOP_LEN * sr)\n",
    "    )\n",
    "\n",
    "    # Fix length (pad or truncate)\n",
    "    if mfcc.shape[1] < FIXED_LEN:\n",
    "        pad_width = FIXED_LEN - mfcc.shape[1]\n",
    "        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')\n",
    "    else:\n",
    "        mfcc = mfcc[:, :FIXED_LEN]\n",
    "\n",
    "    return mfcc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "34c582d1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:57:52.779694Z",
     "iopub.status.busy": "2026-05-25T07:57:52.779198Z",
     "iopub.status.idle": "2026-05-25T08:01:11.000453Z",
     "shell.execute_reply": "2026-05-25T08:01:10.999858Z"
    },
    "papermill": {
     "duration": 198.227324,
     "end_time": "2026-05-25T08:01:11.002727+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:57:52.775403+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train-N: 100%|██████████| 7273/7273 [01:55<00:00, 62.74it/s]\n",
      "train-H: 100%|██████████| 99/99 [00:01<00:00, 77.63it/s]\n",
      "train-M: 100%|██████████| 2062/2062 [00:29<00:00, 70.51it/s]\n",
      "train-L: 100%|██████████| 817/817 [00:11<00:00, 71.77it/s]\n",
      "test-N: 100%|██████████| 2037/2037 [00:30<00:00, 66.48it/s]\n",
      "test-H: 100%|██████████| 59/59 [00:00<00:00, 69.77it/s]\n",
      "test-M: 100%|██████████| 506/506 [00:07<00:00, 67.93it/s]\n",
      "test-L: 100%|██████████| 79/79 [00:01<00:00, 64.55it/s]\n"
     ]
    }
   ],
   "source": [
    "save_mfcc_dataset(ROOT_PATH, \"train\")\n",
    "save_mfcc_dataset(ROOT_PATH, \"test\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1765fe59",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:11.179521Z",
     "iopub.status.busy": "2026-05-25T08:01:11.178949Z",
     "iopub.status.idle": "2026-05-25T08:01:11.185587Z",
     "shell.execute_reply": "2026-05-25T08:01:11.184842Z"
    },
    "papermill": {
     "duration": 0.076752,
     "end_time": "2026-05-25T08:01:11.186938+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:11.110186+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class MFCCDataset(Dataset):\n",
    "    def __init__(self, root_dir):\n",
    "        self.files = []\n",
    "        self.labels = []\n",
    "\n",
    "        for severity in os.listdir(root_dir):\n",
    "            sev_path = os.path.join(root_dir, severity)\n",
    "\n",
    "            if not os.path.isdir(sev_path):\n",
    "                continue\n",
    "\n",
    "            for file in os.listdir(sev_path):\n",
    "                if file.endswith(\".npy\"):\n",
    "                    self.files.append(os.path.join(sev_path, file))\n",
    "                    self.labels.append(label_map[severity])\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.files)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        mfcc = np.load(self.files[idx])\n",
    "\n",
    "        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)\n",
    "        label = torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "\n",
    "        return mfcc, label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ca2cb382",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:11.307462Z",
     "iopub.status.busy": "2026-05-25T08:01:11.306814Z",
     "iopub.status.idle": "2026-05-25T08:01:11.328834Z",
     "shell.execute_reply": "2026-05-25T08:01:11.328235Z"
    },
    "papermill": {
     "duration": 0.083364,
     "end_time": "2026-05-25T08:01:11.330309+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:11.246945+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_dataset = MFCCDataset(\"/kaggle/working/mfcc_features/train\")\n",
    "test_dataset  = MFCCDataset(\"/kaggle/working/mfcc_features/test\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "553a03b1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:11.451986Z",
     "iopub.status.busy": "2026-05-25T08:01:11.451564Z",
     "iopub.status.idle": "2026-05-25T08:01:11.471235Z",
     "shell.execute_reply": "2026-05-25T08:01:11.470508Z"
    },
    "papermill": {
     "duration": 0.081509,
     "end_time": "2026-05-25T08:01:11.472652+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:11.391143+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_size = int(0.1 * len(train_dataset))\n",
    "train_size = len(train_dataset) - val_size\n",
    "\n",
    "train_ds, val_ds = random_split(train_dataset, [train_size, val_size])\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9c8b1471",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:11.592638Z",
     "iopub.status.busy": "2026-05-25T08:01:11.592174Z",
     "iopub.status.idle": "2026-05-25T08:01:11.597971Z",
     "shell.execute_reply": "2026-05-25T08:01:11.597424Z"
    },
    "papermill": {
     "duration": 0.067406,
     "end_time": "2026-05-25T08:01:11.599403+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:11.531997+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "class CNNModel(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CNNModel, self).__init__()\n",
    "\n",
    "        self.features = nn.Sequential(\n",
    "            nn.Conv2d(1, 16, 3, padding=1),\n",
    "            nn.ReLU(),\n",
    "\n",
    "            nn.Conv2d(16, 32, 4, padding=1),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d(2),\n",
    "\n",
    "            nn.Conv2d(32, 64, 5, padding=1),\n",
    "            nn.ReLU(),\n",
    "\n",
    "            nn.Conv2d(64, 128, 2),\n",
    "            nn.ReLU(),\n",
    "            nn.MaxPool2d((1, 2))\n",
    "        )\n",
    "\n",
    "        # Adaptive pooling → avoids manual dimension calc\n",
    "        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 50))\n",
    "\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(128 * 1 * 50, 128),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(128, 64),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(64, NUM_CLASSES)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.features(x)\n",
    "        x = self.adaptive_pool(x)\n",
    "        x = x.view(x.size(0), -1)\n",
    "        x = self.classifier(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e1de2fae",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:11.719022Z",
     "iopub.status.busy": "2026-05-25T08:01:11.718764Z",
     "iopub.status.idle": "2026-05-25T08:01:17.383086Z",
     "shell.execute_reply": "2026-05-25T08:01:17.382423Z"
    },
    "papermill": {
     "duration": 5.726026,
     "end_time": "2026-05-25T08:01:17.384725+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:11.658699+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model = CNNModel().to(device)\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e9c28df1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:17.579732Z",
     "iopub.status.busy": "2026-05-25T08:01:17.578807Z",
     "iopub.status.idle": "2026-05-25T08:01:22.048263Z",
     "shell.execute_reply": "2026-05-25T08:01:22.047358Z"
    },
    "papermill": {
     "duration": 4.532373,
     "end_time": "2026-05-25T08:01:22.049723+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:17.517350+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://download.pytorch.org/whl/cu121\r\n",
      "Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.10.0+cu128)\r\n",
      "Requirement already satisfied: torchvision in /usr/local/lib/python3.12/dist-packages (0.25.0+cu128)\r\n",
      "Requirement already satisfied: torchaudio in /usr/local/lib/python3.12/dist-packages (2.10.0+cu128)\r\n",
      "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.24.3)\r\n",
      "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)\r\n",
      "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)\r\n",
      "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)\r\n",
      "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.1)\r\n",
      "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)\r\n",
      "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2026.2.0)\r\n",
      "Requirement already satisfied: cuda-bindings==12.9.4 in /usr/local/lib/python3.12/dist-packages (from torch) (12.9.4)\r\n",
      "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.93)\r\n",
      "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)\r\n",
      "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)\r\n",
      "Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)\r\n",
      "Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.4.1)\r\n",
      "Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.3.83)\r\n",
      "Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.9.90)\r\n",
      "Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.3.90)\r\n",
      "Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.8.93)\r\n",
      "Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)\r\n",
      "Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.5)\r\n",
      "Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch) (3.4.5)\r\n",
      "Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.90)\r\n",
      "Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch) (12.8.93)\r\n",
      "Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.13.1.3)\r\n",
      "Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.0)\r\n",
      "Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-bindings==12.9.4->torch) (1.3.5)\r\n",
      "Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from torchvision) (2.0.2)\r\n",
      "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.12/dist-packages (from torchvision) (11.3.0)\r\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)\r\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)\r\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a2425e68",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:01:22.181605Z",
     "iopub.status.busy": "2026-05-25T08:01:22.180346Z",
     "iopub.status.idle": "2026-05-25T08:04:06.625885Z",
     "shell.execute_reply": "2026-05-25T08:04:06.625079Z"
    },
    "papermill": {
     "duration": 164.514076,
     "end_time": "2026-05-25T08:04:06.627468+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:01:22.113392+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 33.72it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20: Loss=109.4721, Val Acc=0.9093\n",
      "✅ Best model updated! Val Acc: 0.9093\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 38.98it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/20: Loss=72.2501, Val Acc=0.9141\n",
      "✅ Best model updated! Val Acc: 0.9141\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 38.78it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/20: Loss=54.6724, Val Acc=0.9405\n",
      "✅ Best model updated! Val Acc: 0.9405\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 38.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/20: Loss=44.1032, Val Acc=0.8985\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 38.35it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/20: Loss=34.3612, Val Acc=0.9444\n",
      "✅ Best model updated! Val Acc: 0.9444\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.55it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/20: Loss=30.6734, Val Acc=0.9678\n",
      "✅ Best model updated! Val Acc: 0.9678\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.41it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/20: Loss=23.2076, Val Acc=0.9620\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.73it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/20: Loss=17.1948, Val Acc=0.9649\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.54it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/20: Loss=18.1034, Val Acc=0.9717\n",
      "✅ Best model updated! Val Acc: 0.9717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 36.09it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/20: Loss=9.8384, Val Acc=0.9639\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/20: Loss=8.2602, Val Acc=0.9678\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.96it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/20: Loss=10.6844, Val Acc=0.9649\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.35it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/20: Loss=7.2100, Val Acc=0.9678\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.86it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/20: Loss=10.1320, Val Acc=0.9688\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.84it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/20: Loss=4.1196, Val Acc=0.9707\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.93it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/20: Loss=3.8124, Val Acc=0.9717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.93it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/20: Loss=9.0200, Val Acc=0.9600\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.88it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 18/20: Loss=6.7922, Val Acc=0.9610\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.78it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 19/20: Loss=3.9944, Val Acc=0.9268\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 20/20: Loss=1.0699, Val Acc=0.9707\n",
      "✅ Model copied to working directory!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "best_val_acc = 0\n",
    "save_path = \"/tmp/best_model.pth\"\n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    model.train()\n",
    "    train_loss = 0\n",
    "    for x, y in tqdm(train_loader):\n",
    "        x, y = x.to(device), y.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(x)\n",
    "        loss = criterion(outputs, y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        train_loss += loss.item()\n",
    "\n",
    "    model.eval()\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in val_loader:\n",
    "            x, y = x.to(device), y.to(device)\n",
    "            outputs = model(x)\n",
    "            preds = torch.argmax(outputs, dim=1)\n",
    "            correct += (preds == y).sum().item()\n",
    "            total += y.size(0)\n",
    "\n",
    "    val_acc = correct / total\n",
    "    print(f\"Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}\")\n",
    "\n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)\n",
    "        print(f\"✅ Best model updated! Val Acc: {val_acc:.4f}\")\n",
    "\n",
    "# After training, copy to working directory\n",
    "shutil.copy(save_path, \"./best_model.pth\")\n",
    "print(\"✅ Model copied to working directory!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "62621eb5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:06.862181Z",
     "iopub.status.busy": "2026-05-25T08:04:06.861784Z",
     "iopub.status.idle": "2026-05-25T08:04:06.866763Z",
     "shell.execute_reply": "2026-05-25T08:04:06.866131Z"
    },
    "papermill": {
     "duration": 0.121292,
     "end_time": "2026-05-25T08:04:06.868195+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:06.746903+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def predict(file_path, model_path=\"best_model.pth\"):\n",
    "    model = CNNModel()\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "\n",
    "    mfcc = extract_mfcc(file_path)\n",
    "    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = model(mfcc)\n",
    "        pred = torch.argmax(output, dim=1).item()\n",
    "\n",
    "    inv_map = {v:k for k,v in label_map.items()}\n",
    "    return inv_map[pred]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "bad6aae3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:07.100754Z",
     "iopub.status.busy": "2026-05-25T08:04:07.100012Z",
     "iopub.status.idle": "2026-05-25T08:04:07.104328Z",
     "shell.execute_reply": "2026-05-25T08:04:07.103739Z"
    },
    "papermill": {
     "duration": 0.123223,
     "end_time": "2026-05-25T08:04:07.105758+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:06.982535+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "06d68158",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:07.336544Z",
     "iopub.status.busy": "2026-05-25T08:04:07.335925Z",
     "iopub.status.idle": "2026-05-25T08:04:07.363328Z",
     "shell.execute_reply": "2026-05-25T08:04:07.362727Z"
    },
    "papermill": {
     "duration": 0.143195,
     "end_time": "2026-05-25T08:04:07.364723+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:07.221528+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CNNModel(\n",
       "  (features): Sequential(\n",
       "    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
       "    (1): ReLU()\n",
       "    (2): Conv2d(16, 32, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))\n",
       "    (3): ReLU()\n",
       "    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)\n",
       "    (5): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))\n",
       "    (6): ReLU()\n",
       "    (7): Conv2d(64, 128, kernel_size=(2, 2), stride=(1, 1))\n",
       "    (8): ReLU()\n",
       "    (9): MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False)\n",
       "  )\n",
       "  (adaptive_pool): AdaptiveAvgPool2d(output_size=(1, 50))\n",
       "  (classifier): Sequential(\n",
       "    (0): Linear(in_features=6400, out_features=128, bias=True)\n",
       "    (1): ReLU()\n",
       "    (2): Linear(in_features=128, out_features=64, bias=True)\n",
       "    (3): ReLU()\n",
       "    (4): Linear(in_features=64, out_features=4, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = CNNModel().to(device)\n",
    "model.load_state_dict(torch.load(\"best_model.pth\"))\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ff1f48f5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:07.604999Z",
     "iopub.status.busy": "2026-05-25T08:04:07.603893Z",
     "iopub.status.idle": "2026-05-25T08:04:08.602228Z",
     "shell.execute_reply": "2026-05-25T08:04:08.601247Z"
    },
    "papermill": {
     "duration": 1.121745,
     "end_time": "2026-05-25T08:04:08.604152+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:07.482407+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 84/84 [00:00<00:00, 84.89it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Test Accuracy: 0.9142\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "correct = 0\n",
    "total = 0\n",
    "\n",
    "all_preds = []\n",
    "all_labels = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for x, y in tqdm(test_loader):\n",
    "        x, y = x.to(device), y.to(device)\n",
    "\n",
    "        outputs = model(x)\n",
    "        preds = torch.argmax(outputs, dim=1)\n",
    "\n",
    "        correct += (preds == y).sum().item()\n",
    "        total += y.size(0)\n",
    "\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(y.cpu().numpy())\n",
    "\n",
    "test_acc = correct / total\n",
    "print(f\"✅ Test Accuracy: {test_acc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "63444325",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:08.847159Z",
     "iopub.status.busy": "2026-05-25T08:04:08.846357Z",
     "iopub.status.idle": "2026-05-25T08:04:09.900108Z",
     "shell.execute_reply": "2026-05-25T08:04:09.899167Z"
    },
    "papermill": {
     "duration": 1.177899,
     "end_time": "2026-05-25T08:04:09.901533+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:08.723634+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Classification Report:\n",
      "\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           N       0.99      1.00      0.99      2037\n",
      "           L       0.20      0.37      0.26        79\n",
      "           M       0.81      0.77      0.79       506\n",
      "           H       0.25      0.02      0.03        59\n",
      "\n",
      "    accuracy                           0.91      2681\n",
      "   macro avg       0.56      0.54      0.52      2681\n",
      "weighted avg       0.92      0.91      0.91      2681\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "print(\"\\nClassification Report:\\n\")\n",
    "print(classification_report(all_labels, all_preds, target_names=label_map.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "cf2711dc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:10.136608Z",
     "iopub.status.busy": "2026-05-25T08:04:10.136131Z",
     "iopub.status.idle": "2026-05-25T08:04:12.275260Z",
     "shell.execute_reply": "2026-05-25T08:04:12.274376Z"
    },
    "papermill": {
     "duration": 2.25913,
     "end_time": "2026-05-25T08:04:12.276861+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:10.017731+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 84/84 [00:01<00:00, 83.18it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Saved results.csv and results.xlsx\n",
      "               filename   score_N   score_L   score_M   score_H    prob_N  \\\n",
      "0  M22_Hindi_H_0005.npy -3.348975  1.362809  2.719309 -3.441207  0.001835   \n",
      "1  M22_Hindi_H_0014.npy -3.091048  0.790081 -0.737124  1.803397  0.005167   \n",
      "2  M22_Hindi_H_0037.npy  3.063449  1.180005  0.365000 -4.694145  0.819805   \n",
      "3  M22_Hindi_H_0024.npy -1.884903  2.219005  2.617918 -4.633570  0.006583   \n",
      "4  M22_Hindi_H_0040.npy -2.037250 -1.414209  4.190800 -2.832369  0.001960   \n",
      "\n",
      "     prob_L    prob_M    prob_H predict class actual class  \n",
      "0  0.204091  0.792401  0.001673             M            H  \n",
      "1  0.250470  0.054388  0.689976             H            H  \n",
      "2  0.124664  0.055181  0.000350             N            H  \n",
      "3  0.398761  0.594235  0.000421             M            H  \n",
      "4  0.003655  0.993499  0.000885             M            H  \n"
     ]
    }
   ],
   "source": [
    "# import pandas as pd\n",
    "\n",
    "# inv_map = {v: k for k, v in label_map.items()}\n",
    "\n",
    "# filenames = [os.path.basename(test_dataset.files[i]) for i in range(len(test_dataset))]\n",
    "\n",
    "# scores = []\n",
    "# all_preds = []\n",
    "# all_labels = []\n",
    "\n",
    "# model.eval()\n",
    "# with torch.no_grad():\n",
    "#     for x, y in tqdm(DataLoader(test_dataset, batch_size=32, shuffle=False)):\n",
    "#         x, y = x.to(device), y.to(device)\n",
    "#         outputs = model(x)\n",
    "#         probs = torch.softmax(outputs, dim=1)\n",
    "#         best_scores = probs.max(dim=1).values\n",
    "#         preds = torch.argmax(outputs, dim=1)\n",
    "\n",
    "#         scores.extend(best_scores.cpu().numpy())\n",
    "#         all_preds.extend(preds.cpu().numpy())\n",
    "#         all_labels.extend(y.cpu().numpy())\n",
    "\n",
    "# df = pd.DataFrame({\n",
    "#     \"filename\":      filenames,\n",
    "#     \"score\":         scores,\n",
    "#     \"predict class\": [inv_map[p] for p in all_preds],\n",
    "#     \"actual class\":  [inv_map[l] for l in all_labels],\n",
    "# })\n",
    "\n",
    "# df.to_excel(\"/kaggle/working/results.xlsx\", index=False)\n",
    "# print(\"✅ Saved results.xlsx\")\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "inv_map = {v: k for k, v in label_map.items()}\n",
    "\n",
    "# Class order: Normal → High → Mid → Low → Very_Low\n",
    "class_order = [\"N\", \"L\", \"M\", \"H\"]\n",
    "\n",
    "filenames  = [os.path.basename(test_dataset.files[i]) for i in range(len(test_dataset))]\n",
    "\n",
    "all_preds      = []\n",
    "all_labels     = []\n",
    "all_raw_scores = []   # raw logits per class\n",
    "all_softmax    = []   # softmax probabilities per class\n",
    "\n",
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    for x, y in tqdm(DataLoader(test_dataset, batch_size=32, shuffle=False)):\n",
    "        x, y = x.to(device), y.to(device)\n",
    "        outputs = model(x)                          # raw logits: (batch, NUM_CLASSES)\n",
    "        probs   = torch.softmax(outputs, dim=1)     # softmax probabilities\n",
    "        preds   = torch.argmax(outputs, dim=1)\n",
    "\n",
    "        all_raw_scores.extend(outputs.cpu().numpy())\n",
    "        all_softmax.extend(probs.cpu().numpy())\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(y.cpu().numpy())\n",
    "\n",
    "# Build DataFrame\n",
    "df = pd.DataFrame({\"filename\": filenames})\n",
    "\n",
    "# Raw logit score for each class\n",
    "for cls_name in class_order:\n",
    "    df[f\"score_{cls_name}\"] = [row[label_map[cls_name]] for row in all_raw_scores]\n",
    "\n",
    "# Softmax probability for each class\n",
    "for cls_name in class_order:\n",
    "    df[f\"prob_{cls_name}\"] = [row[label_map[cls_name]] for row in all_softmax]\n",
    "\n",
    "# Final prediction & actual label\n",
    "df[\"predict class\"] = [inv_map[p] for p in all_preds]\n",
    "df[\"actual class\"]  = [inv_map[l] for l in all_labels]\n",
    "\n",
    "# Save both CSV and Excel\n",
    "df.to_csv(\"/kaggle/working/results.csv\",   index=False)\n",
    "df.to_excel(\"/kaggle/working/results.xlsx\", index=False)\n",
    "print(\"✅ Saved results.csv and results.xlsx\")\n",
    "print(df.head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1c192e2b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:12.515404Z",
     "iopub.status.busy": "2026-05-25T08:04:12.514562Z",
     "iopub.status.idle": "2026-05-25T08:04:13.180701Z",
     "shell.execute_reply": "2026-05-25T08:04:13.179919Z"
    },
    "papermill": {
     "duration": 0.788802,
     "end_time": "2026-05-25T08:04:13.182763+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:12.393961+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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r+sck5mZqc2bN9V7/jfffKNXXn5Jv879VZkZGfJ4POrdu4/23Xdf3Xrb7erWrZt9bF5enp5++il9MnOG1qxZo7KyMqWkpGjChL30l7/+TQcffHDQtV9//TX99S8X1LyueQW6//779M7bb2vLls268MKL9NDD/5JUPSvm5Zde1Hvvv6dlS5equLhYXbt21eTJ++jyK65o0SzI6dPPD5pVNOPjj3XWWWeooqJClmXp4osu1JQpU5WSkiKpernyPXdXz8jp269f0EypjRs36oF/3q/vv/9eW7dukWVZ6tq1q/r266e9Juyl6dPP1/ARI3TIIQfpxx9+CKrjr3+5wH49Aq8buIfX8y+8qOSkZD388INavHixXC6XMjKz67yW5RUNL79bu3atbr3lZn333bcqKyvTnmPG6Oabb9VBBwUvRa39vOecc649vuCC6XrzjTckSfvtv7+++uqbOjVI0o8//BB0nR2zw+o7P9DWrVv15BOP68svv9CGDRvk8/mU1r27Jk+erEsuuVQTJuwVdHztP5N5837T/ffdq/fef0/bt21Tr169NW36dF133fVB/y5LSkr06KOPaMbHH2nNmjUqLy9XcnKyevTooXHjx+uE409s12Gmw+GQaZqaOXOGNm7cqH79+kmSXn3lFZWUlEiqDsv8fv9OrxWfkKCrrr56p8e9/PJLQQHhxIkT9d77/wt6L5Cq3w/efON1e5yVlaVLLr7IDghjYmL0zrvv13lfsCxL//vfB0pJCb5eY8L9Pv/qq6/oyquu1qBBg5pcQ6Bly5bqP//5t8466+yQzt+hrKxM7777jj0+4fgTGj2+9p9hakqqHnroQUnVodvq1avqfG4UFhbq2Wef0YyPP9LKlStVVlamtO7ddeCBB+rqq6+p88uH9evX66knn9CC3xdo48aNysvNlc/nU7du3bTnnmN0/vkX6Kijj27Rzy1Vv4bp6en2+L9vvVNnFqVlWfr+++8VHR1d5/ycnBw999yz+vyzT7Vq1SqVlpaqW7duGjlyN5173nk65ZRTg47/9ttv9cLzz2nu3F+VlZUlj8ejQYMG6cijjtall15WJ1wdOnSwNm3cKEm66eZbNHXqVN1911367bf5Kiws1PaMLHt24x9/LNKTTz6hH3/4Udu2pcvpdGrw4ME68aSTdemllyk2Njbo2k39HNnB4/HoiCOP1NtvvSW/368333xDN9zwDwFAZ0ZICACo19NPPRX0BfnxJ56s88VRkqKionT++RfUub8+27al61//erjO/ZmZmfr222/07bffaNas7/X88zVL8eoLUIqKilRUVKT169dpxoyPdfnlV8jr9aq8vFxTpxyoVatWBh2fkZGhjIwMzZ8/X3FxcU0KCX0+n317+7btyszMVGpqatAxqampde6zLEsXX3yhXnn55aD7KyoqtGzZUi1btlTnX3CBHQysWL5cRx19pLZu2RJ0fHp6uj766EN99NGHuvTSy+zQrz7HHHOUfv7ppzr3Z2Vl6eijjtCiRYuC7t++fbs++OB9ffjh//TAAw/q0ssub+SVaLpjjj1Wt99+p/7xj+q9yYqLi/XKKy/ruuvq7lUWKDMzU/vuM0lZWVlB9+9YGjdn9mwNGTIk6Mtdc7322qtBr1FiYmKzzl+zerX223eycnNz7ftm//KLjjn6SL3x5r910kknh1xbuPz444869ZST6ixH3bRxozZt3Kh33n5b9913v6648qp6zy8pLtYB+++nFSuW2/dt2LBet916i8rLy3Xbbbfb959wwnH6YdasoPOzsrKUlZWlP/74Q8VFxe06JDz66GP08ccfye/367lnn9G9990v0zT17LPPSJLS0tI0YMAAzZkzJ2zP+eQTT9i3vV6v/vPft+sEhFL1rLbLLr/CHr/22qsqLCy0x7fddnudgFCSDMPQiSee1KyawvU+3717d23fvl1VVVW666479Oqrrzd4bH1iYmLkdDpVVFSku++6S6eddrqiokJv1PHrr7/ae6OmpKRo8JAhzTq/9pLpbl2D/5zWrF6to446ss7MzS2bN+vNN97Qu++8o5dfeTXofWH5smV68sknVNuO97lPP/1Et9x6m2666eY6xzRH4GeXJC3+Y1GdkNAwDE2ZMqXOufPnz9PJJ52o7du311tjdHR0UEh4/XV/12OPPRp0bGVlpRYuXKiFCxfqtVdf0YyZnzQ4W/ubr7/S/ffdW28Y//xzz+rqq6+q8/MsWrRIixYt0lv//a8++/wLde/eXVLonyMT956ot996q7qeb74mJATQ6RESAgDq9d13NUttk5OTdeyxxzVydNM4HA4NHz5C4yeMV/e07kpMSlJ5ebkWLVyoTz6ZKcuy9Pprr+kvf/mrPePpheefs88fP368jjjyKPl8Pm3Zslnz5s4LCjRmff+9HRB6vV6dd9409ezVSxnbt2vt2rX68cfgGWiN2XPMGHvWz6pVKzVoYH+NGzdOY8eO09hx4zRlylT16tWrznmP/OtfQQHhjqVeqWlpWr16lWbOmGE/5vP5dOqpp9gBodPp1BlnnqlevXprxscfaenSpZKkJ598QnuOGdPg7Jqff/pJe+21l6YedLBKS0rUp08fSdL0aefZAWF8fLxOO+109erdW7N/+UVffvmFTNPU3/9+rcaOG6fJk/dp8mvTmHPPO0833niDPetp1vff7zQk/PB/H9hf7JKTk3XOOeeqS9eu2paerpUrV+rnn2vCvb/+9W868oij7CBSkk455VSNHTtOUsN7Hf7800/q1q2bTjnlVHXp2lXLly1r1s81Z84c9ezZU9de+3cVFRXp1VdfUUVFhUzT1CUXX6SDDz6k2cHjDjv2RXzvvXf022+/SZIGDBiov/71b/YxAwcObPQa+fn5Ov20U+yAMDo6Wuecc67iExL0zjtva9PGjTJNUzfccL3GjB1X7/5sOTk5ysvL05lnnaUePXrq1VdeVnZ2tiTpqSef0D/+caPcbrdWLF9uB4QOh0NnnnWWhgwZqpzsbG3YsEE//DCrzrXbmylTpmjt2jVaunSpXn31Fd18y6365uuvtWHDeknSBX/5a5N/jqLCQj3yr7ohfu8+ve1AJT09XStXrrAfO+SQQ5u8d993335r3zYMQ2cHzExtqXC9z0+aNFk5uTn6YdYsvfP227r22r9r991HNfn86OhoXXTxJbr7rju1YcN6vfTiC7rwootDqkVS0HvGmDFjm3ye3+/X6lWr9NprrwacP0aDBg8OOubUU0+xA8KUlBSddtrpSu7SRV999aXmzJ6tiooKnT99msaMGWv/23W5XBo9erTGjhunbt1SlJCQoJKSEs2e/Ytmff+9JOm+e+/ReedNq/ezpamGDRuu6OholZWVSZKuueZqPfzwQ5o4cZL2HDNGkyZN1uTJk+V0OoPOKyoqqhMQHnjgFE2aPFlFhYX65Zefg47/97/fDAoIR47cTcced5y2bUvXm2+8Ib/fr61bt+q0U0/V7wsXyeWq+7Vzzpw5iomJ0f/93xnq2auXFi1cKKfTqdmzf9GVV15hN53Ze++9dcihh6m4qEhvvvmGsrOztXz5Mp0/fZo++fQzSc37HAk0bvx4+/a8uXNVWVkpt9vd1JcbAHY5hIQAgHqlp2+1bw8ZMiQsewSNGDFSCxf9oU2bNum33+YrY/t2RbmitM8+++r33xdo69bq5/zqq6/skDBwI/2H//VInVmAGzZssP8PfXlFzbH77be/Hn3s8aBjKyoq7NBjZy6//Aq98cbr9rKoqqoqzZkzx55ZZBiGDj/8CD3y6GP2klvTNPXIIzUzJXv16qXZc+YGzTbMycmxvyx9+sknQbMeH3nkUf31bxdKkq6//gaNHj3Kfv7HH3u0wZDw+ONP0H/++1bQn9HixX/oq6++tMfvvveBDjzwwJpzjjtWn3/+mSzL0mOPPhq2kLBLly5KSUmx97sKXPbWkMA/45NOOln/fODBoMdLSkpUXFwsSXboEhgSHnLooUFLeuuTkJCgX2b/qr59+zbtB6klKipK3343y/6znjR5ss479xxJ1QHd+++/p+nTzw/p2jv2RVy6bIkdEvbu07tJS1h3eOP115STk2OP33rrHR12+OGSqv8ujxwxTMXFxbIsS088/liDTRwCZ5buvdfeOuWU6plphYWFWrVqpXbffVTQv7OhQ4fp+edfDFqK7Pf7taXWzNj26OJLLtUlF1+k3Nxc/ec//7aXp7rdbv2lGSFhXl5e0N/HHfbbf/+AkHBr0GPDhg1rcp2B/4ZSU1PDujdeON/n77zzLh14wP4yTVO333ab3nv/g2adf8UVV+rZZ55Wdna27r//Pp1z7nkh17Ju3Vr7dlP2mNu0cWPQ8v4dxo8fr3//562g+z779FMtW1b9Cxyn06nvvptlz1S84YZ/aO+9xmvJkiUqLy/Xs888rQcefEhSzX6iq1et0sJFC5WdlaUoV5QOP+wIzZs7V6WlpfL5fPr+++905plnhfyzx8TE6K677ta1115j35eenq4PPnhfH3zwvqTqmbI33niT/nbhRfYxb7z+WlBAeMedd+n664P32ly3bp19+7FHH7Fv9+vXXz//Mttevjxu7DhdfvllkqTVq1fp008+qdPISqp+/b759rs6Qe6jjzxiB4T7H3CAPv/8S/vv5smnnKJ996ne2/abb77W4sV/aNSoPZr1ORKoV6+avx/l5eVKT0/fpZqwAUBzERICANpMTk6OLjh/uj777NNGjwtcervPPvtq8eLFkqSjjjxCe+89UYMHD9aIESO07377Bc1WGTduvDwejyoqKvTVV19qzJ6jtfuoURoyZIj2HL2npkytf/ZffRITE/Xjjz/r3nvu1ttvv6X8/Pygxy3L0meffap169bp17nz5PV6tWrlyqClThdffGmd5chdu3a1b//6a/BSxjMDQsDo6GidfNLJ9vLsxYsXq7S0tN49wa67/vo6X+5n//JL0Pjwww5p8GedM2d2g4+FormdYSdNnizDMGRZll588QX99tt8DR8xQkOHDtO4seN0wIEHKi0trUU1nXnmWSEHhFL138PAL46nnHKq/nLB+aqqqt7D8PcFC6QQQ8Jw+PXXX+3bKSkpdkAoVQdLhx12uN5//70/j61/Ca3T6dQFf/mrPR46dGjQ43l5+ZKk4cNHqGvXrsrJydGKFcs1csRw7bnnnho8ZIhGjRqlqVMPsvf425n6ZuA112GHHxZS85kzzjhTt9x8k3Jzc3XvPXfbYdyJJ51kL2Hs6Bp6fZsTQDfVxImTdNRRR+uTT2Zq5swZDf49a0h8fLz+/vfrdP3112n79u166qknFVdrz7mmys6q+WVQqKFqamqqbrvtjjp/l2fPrnlv9fv92n33kQ1eI/C9dcOGDTrvvHM0Z3bj77c7flnWEpdedrl69+mjhx96UPPmzavzeEZGhq644nJFx8TYv2D5JeAzIz4+Xtde+/c65+2YFVlaWmp/LkvSSSedFLS/4ZlnnW2HhFL1e059IeFhhx1e70zPwNf4h1mzFBPtafBnnTN7tkaN2iPkz5HAz2RJys7OIiQE0KkREgIA6tWzZy+tWbNakrR69WpZllWnoUhzXfi3v+40IJSkisoK+/add92t9evX64svPldxcbG++eZrffNNTROR/fbfXx9++LFiY2PVu3dvvfDiS7r6qivt5UjLl9csK42Li9PTzzwb1HG3MWlpaXrs8Sf0r0ce1aJFCzV/3jz98MMPmjHjY1VUVNe4cuUKff75Zzr++BOUm5cbdH7/Af0bvX7g8XFxcXU2YU9NrflCY1mW8vPz6w0Jhw0bXs+18+rc15Daezi1RG5ubtBszaYsqZwwYS898MCDuuOO21VcXKzff/9dv//+u/14t27d9O//vBXUhba5mjNzqz4ptcJep9Oprl272jNv8gsK6j2vdmBaUVHZojoaErhXYu1guvZ9tfcs3CEtLU1er9ceuz3BX8x3zOzxer1689//1V//eoE2b9qk9evXaf36mhlGbrdbd911d4N7HwaqbwZec3Xt1jWkkDA6OlrTp5+vhx56MGi23qWXXtbIWXXVbsJTn549g385sXLlygaOrO/cnvb7WGZmpvLy8pScnNykcxt6fXeEhOF+n7/9jjv06aefyLIs3XrrLeqS3LyA7m8XXqTHn3hcW7ds0b8efkiXX3FlyLU0x47uxllZmfr3v99URkaGMjMzdfzxx2rmJ58F7d9X+32+MVkBYeWpp5ykP/74Y6fn7Phsaanjjz9Bxx9/grKysvTrnDma8+sczZwxI2iLjscfe9QOCQPfQ3r37lNnOXKgvLy8oPe2wM8qSYqNjVVcXJw9c6+h95yG3pcDa9mZrD8/b0L9HGnuL7UAYFdHSAgAqNeUKVPsL495eXma8fHH9c4EaKqSkhJ9+uknAdefqqeefkb9+vWT0+nUvvtM0vz58+ucl5CQoI8+nqEtW7Zo7txftXr1ai1fvkwff/SRSktL9eMPP+jhhx/SrbfeJkk69dTTdMIJJ2revLlaumSJ1qxZo1mzvtfChQtVXFysC//2Vx155FGKi4trcu1Op7N6L8Kx4/TXv12on376SQcfVPOlcc2a6oCg9hfiDes3NHrdwOOLi4tVUlISFBRmZmbYtw3DsDs+1lY7XKy+dnCIcOtttyvaW7eTZbi9/tprQV+6Dqxnc/z6XHb5FTr/gr/o119/1fJlS7VmzRp9+eWXWrNmtbKzs3XBBdO1evXanV+oATEhzkjaIevP5dM7+P3+oOW9SQH7Ee6YzSJJ5X/uC7bD2j//TYVb4GypzFq11r6voYDJVatRRGNh0ZQpU7Ry5Wr9/vsCLVq0SOvWrtXsObP1808/qbKyUv/4xw06+uhjgvZya4/+duFFevTRR+zmCHvvvbfGj58Q9ufp2bOnhg0bbu9L+NVXX2rbtm3q0aPHTs+dMnWq/YsRy7L05huvBzU2aYlwv8+PGrWHTj3tNL391lua9f33zZ6R6fV6deONN+mSiy9SXl6envuzkUxzde1WMzusoYAqUGB34/PPv0B77TVeJSUl8vv9uuLyy7Tg94X2NhGB79ter1e33XZHg9fdsUfqqpUrgwLC004/Xffee7969uwpwzDUp3fPsP6yJlBKSoqOPuYYHX3MMbrrrrt11JFH6Ntvq7ui7/jskoLfQ7Zs2Sy/399gUJicnBz0Phf4WSXVXdrb0HtOQ+/LXbp0sd+zJu+zj445+tgGf76Jk2q2IAnlc6R2INmtW0qDzwUAnUHLN5gCAOySLrr44qAvCJdffqn++GNRneOqqqr08ssv1RtMBCooKAjqYHjEEUdo4MCBcjqdWrVyZdDSpUBLly5RVVWVevfurRNPPEnXX3+DXn31dU2bNt0+ZuGfswVyc3O1ceNGRUVFafLkffSXv/5N/3zgQX32ec3efKWlpXW6H9fnsUcf0Vtv/Tdon6Md4uKCv9jsaFgxdNgwpaTUfMF4+pmn6uyBmJeXZ3cqnThxUtBj/37zDft2WVmZ3vtzeagk7bHHHvXOImzIxEnB1+7WtauuuvrqOv8ddvhh2mvvvZp83cZ8+sknuv32W+1xfHx80J9TQ9LT05WRkaGYmBhNmTJFF19yqf71yKN689//to/ZvGlTUCgXuAl+WWlpWOpvzM8//6QNGzbY43fffcdeaixJY8bWLJkLDHN/nVuzDPirL7/UggULGnyOwG6uzf2ZJk6s+aKclZWlLz7/3B5nZmbqiy8+Dzg2+O9Gc5WXl2vF8uVyOBwaN268pk8/X3ffc6++/vpb+9+CaZr6Y/HOZ02VV1S1+L+d7UfZmD59+ui44463xxdfcmnI19qZSy+tuXZ5ebnOOOP0emdM5eXl6YnHH7PH5557nuLj4+3x7bffFtRwZAfLsvS//31g75cnNfz67hDu93lJuvXW2+1/n7W75DbFueeep8GDh4R8vlTd+GeH5u6POWjwYF15Vc1y7FWrVuq///2PPQ7891NeXq6RI0fW+966z777aPyfTTFycmveuyTpxBNPUq9evWQYhmbNmhXWgDA9PV1XXXmFVq5YUecxwzCCPkcC36smT55s3y4qKrK3ugi08c89cmNiYrTHHnvY97///vt2oxQp+LNMav57TuDxGdszdP4FF9R5fS+86CKlpKZo0qTqukP5HJGqA9EdvF5vkxsKAcCuipmEAIB6jRy5m267/Q7desvNkqq/rE2eNFFHHnmURu+5pwzD0Nq1a/T1V18pIyNDU6ce1Oj1UlNTlZSUZO/td//99ykzK0t+n0+vvfZqg0usbrj+es2fP09TpkxV79691S0lRdvS0/X666/ZxyQmVQcTq1ev0gH776fx48dr1B57qEePnnK5XPryyy+CrpmYmLTTn3/xksW6/vrrFB8fr3333U+77b67EhIStH3bNr333rv2cU6nUwcddLCk6k6vV111jW68sXqz961btmjP0aPs7sYbNqzXjI8/1hdffqXRo/fUEUceqaFDh9mh5VVXXan5v81Xz569NOPjj+ymJZKaPXNojz1G66CDDrZnIF155RX64osvNGbsWDkcDm3atFFzZs/RihXLddPNt2ifffZt1vUl6eWXX1JiQqJycrL1008/au7cufZjhmHo2eeeV7du3XZ6nZ9++lHnnXuOJu+zj4YPG64ePXvK7/frow//Zx/jdruDvtz27NXLfn0effRR5eTmKtobrdF77qmpU6c2+2fZmaqqKk2dcoDOOONMu7vxDomJiTrppJPt8bhx4/X1119Jkv7z738rfWu6oqOj7fsaErgkdcGCBbrm6qvUu3cfud1RumQnS2DPOvsc3XffvfYX4NNPP7U6XEpI0Ntvv2XP6jEMw25MEqr8/HztueceGjlyN42fMF49evRUdHS0fvn5ZxUELLtuyr+z9uC++/+p0047XZJ0xJFHttrzTD//As2cOdMObGf/8otGjhim444/XgMHDlJ5ebmWLF6sr7/+Simpqfa/+ZSUFD3x5FOadt65sixLJSUlOvKIwzR16kHae+JEud1ubdq4Ud988402btygL778urEygoT7fV6SBg0apPPOm6YXX3whhFep+hcAt952m845O/TmHTuCI0lauPD3Ro6s36WXXqbHHn3E/nfz0IMP6swzz5LD4dARRx6p4cNH2Mt2TznlJB13/AkaMWKETNPUunVr9dNPP2nTxo16/oUXNXr0nho0aLAcDoe9ZP+aa67WokWLlJuTE/RZFg6VlZV65pmn9cwzT2u33XbTxImT1LtPH/n9fs3+5Zeg7ToOOfRQ+/bZ55yrf/7zfmVkVM8KvOXmm/Tdt99q74kTVVZaql/n/qpuXbvp3feqm59cceVVmj7tPEnSxo0btM/kSUHdjXcYMmRos/9dXXHllZox42NZlqW1a9do7Ng9dfxxxys1LU2FBQVasmSJfvzxB5WUlNgNvUL5HJGkBX82i5Kk8RMm0NkYQKdHSAgAaNB1112v2JgY3XjjP1RRUSGfz6ePP/5IH3/8UbOv5XK5dO3fr9PNN90oqXrW30MPPiBJ2m233dS/f/8GZ1nl5eXZXRlr83q9uqTW7J/58+fXu3RZqt6nadCgQU2uu6ioSJ999mmDeyneetvt9mbuUvVeX6vXrNIrL78sScrOztZzzz1b77kul0vvvPOujjr6SG3dskV+v1+vv1b3C+Mll1zaYGfjxrzy6ms65ugjtWjRIpmmqU8+malPPpnZ7Os05IF/3l/v/V27dtXzz7+oo44+usnXMk1TP/34o3768cd6H7/ooouDNsY//rjj9fifs63Wr1+nO++4XZJ04YUXtUpIOGbMGK1Zs0YPPRTcMdPhcOiJJ5+yZ9BJ0lVXX6NvvvnaXor3/ffVs766du2qgQMH1ttIQJKOPfZY3XfvPTJNU6Zp6qmnnpRUvZx8ZyFhUlKS3nr7XZ1y8onKz89XWVmZnq21VNPhcOjee+9vsLNxcy1btjRo1lqgCRMmhO15Wlvfvn1b1NSmqZxOp/771tu6+uor9eor1SFzfn6+Xnv11Z2ee/rp/yeXy6VLLr5IBQUFsiyrzv6soQrn+/wO/7jxJr355hv1zsRuilNOOVUPPfhAk/bwq8+kSZMUExOj0tJSZWRkaP369RowYECTz09OTtZf/vJXPfJIdeOXlStX6IMP3tfJJ58il8uld999T0cffZQ2btygyspKvfvO241eLzU1Veeff4FeeOF5SdKWzZt17z13S6reemPVqpVhaVhS29KlS7V0af3/Rvv166877rjLHsfHx+v9D/6nk048wQ4Kv/32G3tpsiQdc0zNst8zzjhTixYu1GOPPSqp/veDnj176u133gma+d0U++yzrx599DFdc83V8vl82rJ5s5588omdntfczxFJmhPQYGfHL/wAoDNjuTEAoFGXXHqZVqxcrZtvuVWT99lHKSkpcrlciomJ0fDhI/TXv/5NX371TZO6mV577d/12GOPa8iQoYqKilL37t01ffr5+urrbxXbwB6BV119jS699DLtvffe6tWrl9xutzwejwYMGKizzj5bP/38i72H2NChw/TPfz6g448/QUOGDFViYqKcTqeSk5M1afJkPfzwv/TGm/+u93lqu+ee+/TKq6/p3PPO09ixY9Wrd295PB55PB7169dfp5xyqj7/4itdf/0NQecZhqFnnnlOMz/5TCeddLJ69+kjt9utuLg4DR06TOeff4F69eptHz98xAjNm/ebbr7lVo0ZM0ZxcXFyuVzq0aOHjjvueM2Y+ake/tcjTaq5ttTUVP340y964okndeCBU9StWzc5nU7FxsZq2LDh+r8zztCrr72uq6++JqTrS9VBZ3JysoYNG65jjjlWTz71tFatXtusgHDy5H10x5136YgjjtTAgYMUHx8vl8ullJQUTZkyVS+8+JL++UBwOHfHnXfp0ksvU6/evRvdYD9cdtt9d/300y867rjjlZycrOjoaE2cNEkffTSjTiOcgw46SO+8857GjBkjt9utrl276vT/+z/9MvtXDRtet8nMDqNH76nX33hTY8aMCWog0lT77beffluwUFdeeZVGjtxNMTExcrvd6tO3r07/v//T97N+0JVX7byZyM4kJyfr0Ucf06mnna4RI0aqS5cucjqdSkhI0Lhx43Tb7Xfos8+/bHYw0BnExMTo2Wef169z5+miiy7WHnvsoaSkJDmdTiUmJmr8+PG66eZbNOPjumH+ySefohUrV+v++x/Q1KkHKS0tTW63W16vV4MGDdZZZ5+tDz/8WPvss0+z6wrn+7wk9erVSxdddHGz69jBMIygAKu5YmNjdfIpp9jj/33wQbOvccWVV8kT0LzngX/ebwf/Q4YO1fzfFujee+/XxEmTlJycLKfTqfj4eI0aNUrTpk/XO++8p9NP/z/7/EcefUy33na7+vbrp6ioKPXp21dXX32NPvjfh3KG8d9K37599d33s3Tb7XfooIMO1tChw+z6kpKStNdee+nW227X3Hnz6yytHT9+ghb8vki33Hqbxo8fr4SEBLlcLqWmpurAA6folFNPDTr+nw88qE8+/VwnnHCievbsqaioKMXFxWn06NH6x403ad78BSE1FZKq9wv9de48nX/+BRoyZKhiYmLkcrmUlpam/fbf/8/r18wCDOVzpKKiQp99Wv0LQIfDEdIv4wBgV2OUV1TR0gkAAADALmP+/Hnad5/qZcdjxozR7Dlzd3IGOpuPPvpQp51aHSYfeeRR+uB/H0a2IABoB5hJCAAAAGCXMn78BB155FGSpN9//10///xThCtCe7NjCbNhGLrl1lt3cjQAdA6EhAAAAAB2OXfdfbccjuqvOw89+OBOjkZnMn/+PP34ww+SqpfyjxkzdidnAEDnwHJjAAAAAAAAoJNjJiEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyREStoBlWfL7/bIsK9KlAAAAAAAAACEjJGwB0zS1fNkSmaYZ6VIAAAAAAACAkBESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ0cISEAAAAAAADQyRESAgAAAAAAAJ2cK9IFAAAAAAAAAJZlyefzye/3S7IiXU4HYcjpdMrlcskwjBZdiZAQAAAAAAAAEVVZWant29JVWloS6VI6pJiYWHXv0VNutzvkaxASAgAAAAAAIGJM09SG9WvlcrnUt29fud3uFs+K6ywsy6oOWLdv14b1azV4yDA5HKHtLkhICAAAAAAAgIiprKyUaZrq06ePYmNjI11OhxMTE6OoqCitXbtWlZWV8nq9IV2HxiUAAAAAAACIoOr9B0OdAYfA1y70vRx59QEAAAAAAIBOjpAQAAAAAAAA6OQICQEAAAAAANChbdiwQQ6HQwsXLgzrsc01YMAAPfroo2G/blsgJAQAAAAAAEC7NW3aNDkcDjkcDrndbg0cOFDXXXedysvL7WP69Omj9PR07b777q1WR2FhoW666SaNGDFC0dHR6tGjhw455BB98MEHsqzQ9wJsL+huDAAAAAAAgHbt8MMP18svv6yqqir99ttvOu+882QYhv75z39KkpxOp7p3795qz5+fn6/99ttPBQUFuuuuuzRhwgS5XC7NmjVL119/vaZOnaqkpKRWe/62wExCAAAAAAAAtGsej0fdu3dXnz59dPzxx+vggw/W119/bT9eewlxXl6ezjrrLKWmpiomJkZDhw7VK6+8Uu+1/X6/pk+frhEjRmjTpk31HnPjjTdqw4YNmjNnjs4991yNHDlSQ4cO1V/+8hf9/vvviouLq/e8f/3rX9pjjz0UFxenvn376uKLL1ZxcbH9+MaNG3XssceqS5cuiouL0+67765PP/202T9DODCTEAAAAAAAAB3GkiVL9Msvv6hfv34NHnPLLbdo2bJl+vTTT9WtWzetWbNGZWVldY6rqKjQGWecoQ0bNuiHH35QSkpKnWNM09Tbb7+tM844Qz179qzzeEMBoSQ5HA499thjGjBggNatW6dLLrlE1113nZ5++mlJ0qWXXqrKykrNmjVLsbGxWrZsmX29pv4M4UJICAAAAAAAgHZt5syZio+Pl8/nU0VFhRwOh5544okGj9+8ebP23HNPjR8/XpLUv3//OscUFxfr6KOPVkVFhb799lslJibWe63s7Gzl5eVp+PDhza77yiuvtG/3799fd911ly666CI7JNy0aZNOPPFEjRo1SpI0cODAZv0M4URICAAAAAAAgHZtypQpevrpp1VSUqJHH31UTqdTJ510UoPHX3jhhTr55JP1+++/65BDDtHxxx+vyZMnBx1zxhlnqHfv3vrmm28UHR3d4LVa0pTk66+/1v33368VK1aosLBQPp9P5eXlKi0tVUxMjC677DJdfPHF+uqrr3TQQQfppJNO0h577NHknyGc2JMQANDuWH6fyhZ/qYJPH1LxnLdklhVFuiQAAAAAERQbG6vBgwdr9OjReumllzR37ly99NJLDR5/xBFHaMOGDbryyiu1bds2HXzwwbr22mvrHPPHH39o9uzZjT53SkqKkpKStGLFimbVvGHDBh1zzDEaNWqU3nvvPc2fP19PPvmkJKmyslKSdMEFF2jt2rU666yztGTJEk2YMMGeIdmUnyGcCAkBAO2KWVao4h9fUcWa2bIqSuTbtlJF3z8vX+6WSJcGAAAAoB1wOBz6xz/+oVtuuaXRPfpSUlJ07rnn6o033tAjjzyiF154Iejxiy66SPfdd5+OO+44zZo1q9HnO+200/Sf//xH6enpdR4vLi6Wz+erc/9vv/0m0zT18MMPa+LEiRo6dGi95/fp00cXXnih3n//fV199dV68cUXm/wzhBMhIQCg3bAsSyVz35M/L/iD0yovVskv/5ZZURKhygAAAAC0J6eccoqcTqeeeuqpeh+/9dZb9dFHH2nNmjVaunSpPvnkE40YMaLOcZdddpnuuusuHXPMMfrpp58afL577rlHffr00cSJE/X6669r2bJlWr16tV5++WWNHTs2qGPxDoMHD1ZVVZWeeOIJrVu3Tm+88Yaee+65oGOuvPJKffHFF1q/fr0WLFig77//3q6zqT9DuBASAgDaDX/uFvlzN9f7mFVVrsr189u4IgAAAADtkcvl0iWXXKIHH3xQJSV1JxO43W7deOONGj16tA444AA5nU7997//rfdaV155pW6//XYdddRR+uWXX+o9pkuXLpo9e7bOPPNM3XPPPRo7dqz2339/vfXWW3rggQfqbXoyevRoPfzww3rggQc0atQo/ec//9G9994bdIzf79ell16qkSNH6ogjjtDQoUPt4LM5P0M4GOUVVaHvvtjJ+f1+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6055PVFB/+0xavfW/BEAoEPxZW0IGrtS+kekjtZgRAeHhGZxToQqAQAAAID2oUOEhO+++46uu+7vuummmzXn17kaNWoPHXP0UcrMzKz3+NLSUg0YMEB3332Punfv3uB1R47cTRs2brb/+/a771vpJwCAjseXtb5mYDjlTNg1lhpLkuGOlpxR9piQEAAAAEBn54p0AU3x+GOPavr083XuuedJkp586ml9/vlneu21V/X3v19X5/jx4ydo/PgJkqSbb76pweu6XM5GQ8TaKioqVFFRYY9N02zyuQDQ0QSGhI6EVBnODvGR0SSGYcgRnSizOFuS5CckBAAAANDJtfuZhJWVlVqwYIGmTj3Ivs/hcGjK1Kn6dc6cFl17zZo1GtC/r4YPG6pzzz1bmzZtavT4Bx74p1JTutr/DRrYv0XPDwDtlVlaILOkZp8+Z5deEaymdQTuS8hMQgAAAACdXbsPCbOzs+X3+5Walhp0f1pqmjIytod83b0m7KUXXnxJH8+YqSeeeFIbN2zQQQdNUVFRUYPnXHfd9crMyrH/W7tuQ8jPDwDtWVXgUmNJruRdLyQ0ohPs22ZpvizTH8FqAAAAACCydp21Y8102OGH27dHjdpDE/baS0OHDNJ7772radOm13uOx+ORx+Oxx34/XygB7JqC9iN0RskR1y1yxbQSR2DzEsuSWZInZ/yu93MCAAAAQFO0+5mE3bp1k9PpVGZGcJOSjMwMpaU1fT/BnUlKStKQIUO0du3asF0TADoqX/YG+7YzMU2Go91/XDRb4HJjiSXHAAAAADq3dv+tz+12a+zYsfruu2/t+0zT1Pfffae9J04M2/MUFxdr3bp16tGMRiYAsCsyy4tllRXaY0dijwhW03qCZhKK5iUAAAAAOrcOsdz48iuu1AXnT9fYceM0YfwEPfHE4yopKdE555wrSZo+/Tz17NlLd999j6TqZifLly+TJFVVVio9PV2LFi1UXGycBg0eLEm64frrdORRR6tv377ati1dd915p5xOp0497fTI/JAA0E7487cFjZ0JKRGqpHUZUR4ZUV5ZVeWSmEkIAAAAoHPrECHhKaecquysLN155x3K2L5do0eP1sczZiotLU2StHnzZjkClsKlp6dr770m2ONHHvmXHnnkX9pv//311VffSJK2bt2qc885Szk5OUpJSdHkyfto1g8/KSVl1/wy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fkVTAVUFNBWCpiL5s0pepkdA42kYXkM07Xr6UgQYdX1kOYKJfppp2wT+xImIiCj7ZM3rk4sNRAAs5gtbhm6uyfokYTYxGo0wGlONPBRl7pXlvveqd2Q6BCIiIiIiIiIimmFZX1+Xl5cHSZLQ29Obtr2ntweFhUUT3qawqAi9Y2YE9vT2onB4duHI7Xp7xx7TkzyGiIiIiIiIiIhovsj6JKHBYMCaNWuwbdvW5DZVVfHStm3YuGnThLfZtHETtm3blrZt64svYOPGxPHV1dUoKirCtq2pY3w+H17fs+es90lERERERERERDRXzYpy40/d+2ncc/ddWLN2LdavW4+HH/4BgsEgPvCBDwIA7rrrQygpKcVXv/o1AMDHP/EJXH/dtfjP730PN998Mx77w2PYu3cvfvTjnwBIrM/1iU9+Ct/85tdRV1eHquoqPHD//SguLsGtt96Wse+TiIiIiIiIiIgoE2ZFkvCOO+5Ef18fHnzwAfR0d2PlypV4/Iknk6XBbW1tEMXUpMjLLtuMX/36Udz/la/gy1/+EurqFuAPf/gTli5dljzmc5/7FwSDQXz84x+Fx+PB5s1b8MQTT8Jkmp+LUxIRERERERER0fwlRKLxrGk4M9soioLjx45g8ZJlkKSJ+68SERERERERERFlu6xfk5CIiIiIiIiIiIimF5OERERERERERERE8xyThERERERERERERPMck4RERERERERERETzHJOERERERERERERE8xyThERERERERERERPMck4RERERERERERETzHJOERERERERERERE8xyThERERERERERERPMck4RERERERERERETzHJOERERERERERERE8xyThERERERERERERPMck4RERERERERERETznC7TAcxmmqYBABRFyXAkREREREREREREZyeKIgRBOOt+JgknQVVVAMCpk8czHAkREREREREREdHZLV6yDJIknXW/EInGtRmMZ05RVRWyLJ83E0vTz+/3o7amCg2NzbDb7ZkOh+iS8VymuYTnM80VPJdpLuH5THMFz2WaS2bqfOZMwmkkiiIMBkOmwyAkfhfBYBCiKJ4zK06U7Xgu01zC85nmCp7LNJfwfKa5gucyzSXZcj6zcQkREREREREREdE8xyQhERERERERERHRPMckIc0JRqMRX/zS/wej0ZjpUIgmhecyzSU8n2mu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A9i69UV85CP/eMn3cc89H8Ejj/wMfr///AcTERERjaHLdABEREREc8mNN96ItWvXAQDuuutu5OXm4bvf/Q6efPIJvPOdd2Q4upmn0+mg0136S85f/uIXaGttxbaXtuOyyzan7fP5fDAYDJMN8YIFg0FYrdZpue9f/+pXKK+owMaNmy75Pm5/29vxmc98Gn/60x/xoQ99eAqjIyIiovmAMwmJiIiIptGWLZcDABobG5PbYrEYHnjgfly2aQMK8nOR43bimmvehJdeeinttps2rse77kxPLK5dswomox6HDx9KbvvDHx6DyajHiePHJx1vb28v/umfPoKK8lI4HTasX7cGjz7663HHDQwM4MMf/iDy83JQWJCHu+/+MA4dOgiTUY9f//pXyePGrkloMuoRDAbxm0cfTZZm33PPXWeNp7GxAZIkTZg8czgcMJlMadv27NmN2259K4oK85HjdmLd2tX44cM/SDtm27ZtuOaaNyHH7URhQR7e+Y63j/vZjcR9/PgxfOAD70dRYT6uufqq5P7//d/f4rJNG+By2lFcVID3v+8f0NbWdtbv43wef+JxvOlNb4IgCGnbNU3DN77xddTWVMHtcuCGG67DsWNHUV9fN+7nVlBQgOXLl+PJJ5645DiIiIho/mKSkIiIiGgatbQ0AwDcLldym8/nwy9/8XNceeVV+NrXvo4vfenL6O/rx1vfcgsOHjyQPG7Llsvx6qs7k+PBwUEcO3YMoihi5yuvJLfvfOUV5OfnY9HixZOKNRwO44brr8P//va3ePe734NvfOObcDid+Mg9d6cl2lRVxTvefjse+/3v8b73vR8PPPAguru6cc/dZ0/2jfj5L34Jo9GILZdfjp//4pf4+S9+iXvu+chZj6+orISiKPjtb39z3vt+4YUXcN211+D4ieP4+Cc+iW9969u46qo34amnnkoe8+KLL+Ktb7kFfb19+NKXvoxP3ftp7Nr1Gq6++io0NzePu8/3vuc9CIdCePDBh3DXXXcDAL75zW/g7rs+jLq6Bfj2t7+DT3zyU9i2bSuuu/YaeDye88Y5VkdHB9paW7F61epx+x544H48cP9XsHz5CnzjG99EdXU13vLmWxAKBie8r9Vr1mDXrtcuOgYiIiIilhsTERERTSGv14f+/n5EIouhc/QAAArtSURBVBG8/voefO1rX4XRaMTNt7w5eYzb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\n",
      "text/plain": [
       "<Figure size 1300x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Saved distribution_plot.png\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from scipy.stats import gaussian_kde\n",
    "\n",
    "# Class order: Normal → High → Mid → Low → Very_Low\n",
    "class_order = [\"N\", \"L\", \"M\", \"H\"]\n",
    "colors = {\n",
    "    \"N\":   \"#7ba7d4\",\n",
    "    \"L\":     \"#e8a87c\",\n",
    "    \"M\":      \"#6bb89e\",\n",
    "    \"H\":      \"#e0788a\",\n",
    "   \n",
    "}\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(13, 6))\n",
    "fig.patch.set_facecolor(\"#f8f9fb\")\n",
    "ax.set_facecolor(\"#f8f9fb\")\n",
    "\n",
    "# Compute x-axis range from real scores\n",
    "all_scores_flat = []\n",
    "for cls_name in class_order:\n",
    "    col = f\"score_{cls_name}\"\n",
    "    cls_rows = df[df[\"actual class\"] == cls_name][col].values\n",
    "    all_scores_flat.extend(cls_rows)\n",
    "\n",
    "x_min = min(all_scores_flat) - 0.5\n",
    "x_max = max(all_scores_flat) + 0.5\n",
    "x = np.linspace(x_min, x_max, 600)\n",
    "\n",
    "# Plot each class using REAL scores from df\n",
    "for cls_name in class_order:\n",
    "    col = f\"score_{cls_name}\"\n",
    "    cls_scores = df[df[\"actual class\"] == cls_name][col].values\n",
    "    if len(cls_scores) < 2:\n",
    "        continue\n",
    "    kde = gaussian_kde(cls_scores, bw_method=0.35)\n",
    "    y_vals = kde(x)\n",
    "    ax.plot(x, y_vals, color=colors[cls_name], linewidth=2.5, label=cls_name, zorder=3)\n",
    "    ax.fill_between(x, y_vals, alpha=0.20, color=colors[cls_name], zorder=2)\n",
    "\n",
    "# # Decision boundary at logit = 0\n",
    "# ax.axvline(0, color=\"#555\", linewidth=1.3, linestyle=\"--\", alpha=0.6, zorder=4)\n",
    "# ax.text(0.05, ax.get_ylim()[1] * 0.95, \"Decision\\nBoundary\",\n",
    "        # fontsize=8, color=\"#444\",\n",
    "        # bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"white\", ec=\"#aaa\", alpha=0.8))\n",
    "\n",
    "ax.set_xlabel(\"Raw Logit Score  (q)\", fontsize=12, labelpad=8)\n",
    "ax.set_ylabel(\"Density  p(q)\", fontsize=12, labelpad=8)\n",
    "ax.set_title(\"Class Score Distributions — MFCC-CNN (Real Scores)\",\n",
    "             fontsize=14, fontweight=\"bold\", pad=12)\n",
    "ax.legend(title=\"Risk Class\", fontsize=10, title_fontsize=10,\n",
    "          framealpha=0.9, edgecolor=\"#ccc\")\n",
    "ax.spines[[\"top\", \"right\"]].set_visible(False)\n",
    "ax.spines[[\"left\", \"bottom\"]].set_color(\"#ccc\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"/kaggle/working/distribution_plot.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "print(\"✅ Saved distribution_plot.png\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8dc442a0",
   "metadata": {
    "papermill": {
     "duration": 0.121444,
     "end_time": "2026-05-25T08:04:13.421841+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:13.300397+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "for single file usage "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c0656308",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T08:04:13.663970Z",
     "iopub.status.busy": "2026-05-25T08:04:13.663124Z",
     "iopub.status.idle": "2026-05-25T08:04:13.666762Z",
     "shell.execute_reply": "2026-05-25T08:04:13.666150Z"
    },
    "papermill": {
     "duration": 0.12805,
     "end_time": "2026-05-25T08:04:13.668081+00:00",
     "exception": false,
     "start_time": "2026-05-25T08:04:13.540031+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# def predict(file_path, model, device):\n",
    "#     model.eval()\n",
    "\n",
    "#     mfcc = extract_mfcc(file_path)\n",
    "#     mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)\n",
    "\n",
    "#     with torch.no_grad():\n",
    "#         output = model(mfcc)\n",
    "#         pred = torch.argmax(output, dim=1).item()\n",
    "\n",
    "#     inv_map = {v:k for k,v in label_map.items()}\n",
    "#     return inv_map[pred]\n",
    "\n",
    "# file = \"/kaggle/input/YOUR_DATASET/test/Low/sample.wav\"\n",
    "\n",
    "# print(\"Prediction:\", predict(file, model, device))"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "databundleVersionId": 17443133,
     "datasetId": 10540059,
     "sourceId": 16444587,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31329,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 391.300645,
   "end_time": "2026-05-25T08:04:16.993868+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-25T07:57:45.693223+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
