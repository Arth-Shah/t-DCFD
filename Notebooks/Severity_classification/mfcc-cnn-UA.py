{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d0ad58d0",
   "metadata": {
    "papermill": {
     "duration": 0.004374,
     "end_time": "2026-05-25T10:39:00.381694+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:00.377320+00:00",
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
   "id": "dbf03df3",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:00.389728Z",
     "iopub.status.busy": "2026-05-25T10:39:00.388989Z",
     "iopub.status.idle": "2026-05-25T10:39:05.080259Z",
     "shell.execute_reply": "2026-05-25T10:39:05.079327Z"
    },
    "papermill": {
     "duration": 4.697208,
     "end_time": "2026-05-25T10:39:05.082178+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:00.384970+00:00",
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
   "id": "f4ff9922",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:05.090423Z",
     "iopub.status.busy": "2026-05-25T10:39:05.089720Z",
     "iopub.status.idle": "2026-05-25T10:39:05.093924Z",
     "shell.execute_reply": "2026-05-25T10:39:05.093417Z"
    },
    "papermill": {
     "duration": 0.009824,
     "end_time": "2026-05-25T10:39:05.095248+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:05.085424+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ROOT_PATH = \"/kaggle/input/datasets/arthjs/dataset-ua-asr/dataset_UA_ASR\"  # change this\n",
    "\n",
    "SR = 16000\n",
    "N_MFCC = 20\n",
    "FRAME_LEN = 0.025  # 25 ms\n",
    "HOP_LEN = 0.010    # 10 ms\n",
    "\n",
    "FIXED_LEN = 250  # time frames (important for CNN)\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 20\n",
    "LR = 1e-3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "29251ab0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:05.102663Z",
     "iopub.status.busy": "2026-05-25T10:39:05.102284Z",
     "iopub.status.idle": "2026-05-25T10:39:05.106360Z",
     "shell.execute_reply": "2026-05-25T10:39:05.105841Z"
    },
    "papermill": {
     "duration": 0.00936,
     "end_time": "2026-05-25T10:39:05.107708+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:05.098348+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "label_map = {\n",
    "    \"Normal\":   0,\n",
    "    \"High\":     1,\n",
    "    \"Mid\":      2,\n",
    "    \"Low\":      3,\n",
    "    \"Very_Low\": 4\n",
    "}\n",
    "NUM_CLASSES = len(label_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "141de2d3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:05.115902Z",
     "iopub.status.busy": "2026-05-25T10:39:05.115549Z",
     "iopub.status.idle": "2026-05-25T10:39:05.121592Z",
     "shell.execute_reply": "2026-05-25T10:39:05.120827Z"
    },
    "papermill": {
     "duration": 0.01126,
     "end_time": "2026-05-25T10:39:05.123062+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:05.111802+00:00",
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
   "id": "06864e9c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:05.130263Z",
     "iopub.status.busy": "2026-05-25T10:39:05.129849Z",
     "iopub.status.idle": "2026-05-25T10:39:05.134625Z",
     "shell.execute_reply": "2026-05-25T10:39:05.134019Z"
    },
    "papermill": {
     "duration": 0.009843,
     "end_time": "2026-05-25T10:39:05.136003+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:05.126160+00:00",
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
   "id": "944daffc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:05.143706Z",
     "iopub.status.busy": "2026-05-25T10:39:05.143145Z",
     "iopub.status.idle": "2026-05-25T10:39:07.243971Z",
     "shell.execute_reply": "2026-05-25T10:39:07.243050Z"
    },
    "papermill": {
     "duration": 2.106353,
     "end_time": "2026-05-25T10:39:07.245447+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:05.139094+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train-Very_Low: 100%|██████████| 6820/6820 [00:00<00:00, 3215507.34it/s]\n",
      "train-Mid: 100%|██████████| 6174/6174 [00:00<00:00, 2511944.21it/s]\n",
      "train-Low: 100%|██████████| 6200/6200 [00:00<00:00, 2329125.37it/s]\n",
      "train-Normal: 100%|██████████| 46410/46410 [00:00<00:00, 3618911.83it/s]\n",
      "train-High: 100%|██████████| 10850/10850 [00:00<00:00, 3512790.30it/s]\n",
      "test-Very_Low: 100%|██████████| 3086/3086 [00:00<00:00, 2276001.78it/s]\n",
      "test-Mid: 100%|██████████| 3100/3100 [00:00<00:00, 2800418.35it/s]\n",
      "test-Low: 100%|██████████| 3100/3100 [00:00<00:00, 2546981.86it/s]\n",
      "test-Normal: 100%|██████████| 23205/23205 [00:00<00:00, 3394323.23it/s]\n",
      "test-High: 100%|██████████| 5425/5425 [00:00<00:00, 1788984.92it/s]\n"
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
   "id": "654ce6a8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:07.256829Z",
     "iopub.status.busy": "2026-05-25T10:39:07.256162Z",
     "iopub.status.idle": "2026-05-25T10:39:07.262992Z",
     "shell.execute_reply": "2026-05-25T10:39:07.262098Z"
    },
    "papermill": {
     "duration": 0.0141,
     "end_time": "2026-05-25T10:39:07.264469+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:07.250369+00:00",
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
   "id": "93719833",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:07.275327Z",
     "iopub.status.busy": "2026-05-25T10:39:07.274936Z",
     "iopub.status.idle": "2026-05-25T10:39:07.279213Z",
     "shell.execute_reply": "2026-05-25T10:39:07.278649Z"
    },
    "papermill": {
     "duration": 0.011644,
     "end_time": "2026-05-25T10:39:07.280780+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:39:07.269136+00:00",
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
   "id": "ef1b8354",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:39:07.291825Z",
     "iopub.status.busy": "2026-05-25T10:39:07.291495Z",
     "iopub.status.idle": "2026-05-25T10:39:07.319141Z",
     "shell.execute_reply": "2026-05-25T10:39:07.318047Z"
    },
    "papermill": {
     "duration": 0.034721,
     "end_time": "2026-05-25T10:39:07.320526+00:00",
     "exception": true,
     "start_time": "2026-05-25T10:39:07.285805+00:00",
     "status": "failed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "num_samples should be a positive integer value, but got num_samples=0",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m/tmp/ipykernel_23/803511446.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mtrain_ds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mval_ds\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrandom_split\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_dataset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mtrain_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mval_size\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m \u001b[0mtrain_loader\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mDataLoader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_ds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m \u001b[0mval_loader\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mDataLoader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mval_ds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device, in_order)\u001b[0m\n\u001b[1;32m    392\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# map-style\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    393\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mshuffle\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 394\u001b[0;31m                     \u001b[0msampler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mRandomSampler\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgenerator\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mgenerator\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[arg-type]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    395\u001b[0m                 \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    396\u001b[0m                     \u001b[0msampler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSequentialSampler\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[arg-type]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/torch/utils/data/sampler.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, data_source, replacement, num_samples, generator)\u001b[0m\n\u001b[1;32m    147\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    148\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnum_samples\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mint\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnum_samples\u001b[0m \u001b[0;34m<=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 149\u001b[0;31m             raise ValueError(\n\u001b[0m\u001b[1;32m    150\u001b[0m                 \u001b[0;34mf\"num_samples should be a positive integer value, but got num_samples={self.num_samples}\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    151\u001b[0m             )\n",
      "\u001b[0;31mValueError\u001b[0m: num_samples should be a positive integer value, but got num_samples=0"
     ]
    }
   ],
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
   "execution_count": null,
   "id": "24549499",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:14.450790Z",
     "iopub.status.busy": "2026-05-25T07:46:14.450545Z",
     "iopub.status.idle": "2026-05-25T07:46:14.460343Z",
     "shell.execute_reply": "2026-05-25T07:46:14.459163Z",
     "shell.execute_reply.started": "2026-05-25T07:46:14.450764Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
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
   "execution_count": null,
   "id": "3496c026",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:14.461556Z",
     "iopub.status.busy": "2026-05-25T07:46:14.461297Z",
     "iopub.status.idle": "2026-05-25T07:46:18.123325Z",
     "shell.execute_reply": "2026-05-25T07:46:18.122750Z",
     "shell.execute_reply.started": "2026-05-25T07:46:14.461530Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
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
   "execution_count": null,
   "id": "05a44d68",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:18.124536Z",
     "iopub.status.busy": "2026-05-25T07:46:18.124212Z",
     "iopub.status.idle": "2026-05-25T07:46:22.483245Z",
     "shell.execute_reply": "2026-05-25T07:46:22.482449Z",
     "shell.execute_reply.started": "2026-05-25T07:46:18.124514Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40212aff",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:22.484689Z",
     "iopub.status.busy": "2026-05-25T07:46:22.484430Z",
     "iopub.status.idle": "2026-05-25T07:46:31.669720Z",
     "shell.execute_reply": "2026-05-25T07:46:31.668618Z",
     "shell.execute_reply.started": "2026-05-25T07:46:22.484662Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
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
   "execution_count": null,
   "id": "9cef4654",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:31.673420Z",
     "iopub.status.busy": "2026-05-25T07:46:31.673099Z",
     "iopub.status.idle": "2026-05-25T07:46:31.678491Z",
     "shell.execute_reply": "2026-05-25T07:46:31.677898Z",
     "shell.execute_reply.started": "2026-05-25T07:46:31.673394Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
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
   "execution_count": null,
   "id": "6d8fbac4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:31.679968Z",
     "iopub.status.busy": "2026-05-25T07:46:31.679560Z",
     "iopub.status.idle": "2026-05-25T07:46:31.691033Z",
     "shell.execute_reply": "2026-05-25T07:46:31.690311Z",
     "shell.execute_reply.started": "2026-05-25T07:46:31.679932Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
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
   "execution_count": null,
   "id": "ff89f4b4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:31.692112Z",
     "iopub.status.busy": "2026-05-25T07:46:31.691886Z",
     "iopub.status.idle": "2026-05-25T07:46:31.729274Z",
     "shell.execute_reply": "2026-05-25T07:46:31.728516Z",
     "shell.execute_reply.started": "2026-05-25T07:46:31.692092Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = CNNModel().to(device)\n",
    "model.load_state_dict(torch.load(\"best_model.pth\"))\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fca5589",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:31.730428Z",
     "iopub.status.busy": "2026-05-25T07:46:31.730142Z",
     "iopub.status.idle": "2026-05-25T07:46:32.733739Z",
     "shell.execute_reply": "2026-05-25T07:46:32.732871Z",
     "shell.execute_reply.started": "2026-05-25T07:46:31.730406Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
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
   "execution_count": null,
   "id": "6a212919",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:32.735132Z",
     "iopub.status.busy": "2026-05-25T07:46:32.734820Z",
     "iopub.status.idle": "2026-05-25T07:46:33.231118Z",
     "shell.execute_reply": "2026-05-25T07:46:33.230205Z",
     "shell.execute_reply.started": "2026-05-25T07:46:32.735100Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "print(\"\\nClassification Report:\\n\")\n",
    "print(classification_report(all_labels, all_preds, target_names=label_map.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1084b49",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:11.954649Z",
     "iopub.status.busy": "2026-05-25T07:55:11.954092Z",
     "iopub.status.idle": "2026-05-25T07:55:13.441766Z",
     "shell.execute_reply": "2026-05-25T07:55:13.441039Z",
     "shell.execute_reply.started": "2026-05-25T07:55:11.954564Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
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
    "class_order = [\"Normal\", \"High\", \"Mid\", \"Low\", \"Very_Low\"]\n",
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
   "execution_count": null,
   "id": "df84228b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:56:06.942807Z",
     "iopub.status.busy": "2026-05-25T07:56:06.942198Z",
     "iopub.status.idle": "2026-05-25T07:56:07.517909Z",
     "shell.execute_reply": "2026-05-25T07:56:07.517112Z",
     "shell.execute_reply.started": "2026-05-25T07:56:06.942773Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from scipy.stats import gaussian_kde\n",
    "\n",
    "# Class order: Normal → High → Mid → Low → Very_Low\n",
    "class_order = [\"Normal\", \"High\", \"Mid\", \"Low\", \"Very_Low\"]\n",
    "colors = {\n",
    "    \"Normal\":   \"#7ba7d4\",\n",
    "    \"High\":     \"#e8a87c\",\n",
    "    \"Mid\":      \"#6bb89e\",\n",
    "    \"Low\":      \"#e07b8a\",\n",
    "    \"Very_Low\": \"#9b8ec4\",\n",
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
   "id": "c39d4612",
   "metadata": {
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
    },
    "tags": []
   },
   "source": [
    "for single file usage "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e4864c3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:46:35.316110Z",
     "iopub.status.busy": "2026-05-25T07:46:35.315653Z",
     "iopub.status.idle": "2026-05-25T07:46:35.319994Z",
     "shell.execute_reply": "2026-05-25T07:46:35.319089Z",
     "shell.execute_reply.started": "2026-05-25T07:46:35.316083Z"
    },
    "papermill": {
     "duration": null,
     "end_time": null,
     "exception": null,
     "start_time": null,
     "status": "pending"
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
     "databundleVersionId": 16218042,
     "datasetId": 9794004,
     "sourceId": 15312754,
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
   "duration": 12.115317,
   "end_time": "2026-05-25T10:39:09.844403+00:00",
   "environment_variables": {},
   "exception": true,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-25T10:38:57.729086+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
