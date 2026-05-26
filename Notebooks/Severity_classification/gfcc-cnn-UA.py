{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3c9af82c",
   "metadata": {
    "papermill": {
     "duration": 0.005527,
     "end_time": "2026-05-25T10:34:47.266909+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:47.261382+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# GFCC-CNN: Gammatone Frequency Cepstral Coefficients + CNN\n",
    "\n",
    "Converted from MFCC-CNN. Uses gammatone filterbank (GFCC) instead of MFCC for feature extraction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7b24da82",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:47.277016Z",
     "iopub.status.busy": "2026-05-25T10:34:47.276159Z",
     "iopub.status.idle": "2026-05-25T10:34:57.156012Z",
     "shell.execute_reply": "2026-05-25T10:34:57.155242Z"
    },
    "papermill": {
     "duration": 9.886974,
     "end_time": "2026-05-25T10:34:57.157982+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:47.271008+00:00",
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
    "from tqdm import tqdm\n",
    "from scipy.fftpack import dct\n",
    "from scipy.signal import gammatone\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "91bc38fb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:57.167768Z",
     "iopub.status.busy": "2026-05-25T10:34:57.167367Z",
     "iopub.status.idle": "2026-05-25T10:34:57.171842Z",
     "shell.execute_reply": "2026-05-25T10:34:57.171215Z"
    },
    "papermill": {
     "duration": 0.010622,
     "end_time": "2026-05-25T10:34:57.173141+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.162519+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ROOT_PATH = \"/kaggle/input/datasets/arthjs/dataset-ua-asr/dataset_UA_ASR\"  # change this\n",
    "\n",
    "SR = 16000\n",
    "N_GFCC = 20           # number of GFCC coefficients (same as N_MFCC was)\n",
    "N_FILTERS = 64        # number of gammatone filters\n",
    "FRAME_LEN = 0.025     # 25 ms\n",
    "HOP_LEN = 0.010       # 10 ms\n",
    "LOW_FREQ = 100.0      # lowest center frequency for gammatone bank\n",
    "\n",
    "FIXED_LEN = 250      # time frames (important for CNN)\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 20\n",
    "LR = 1e-3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9f044fb1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:57.182803Z",
     "iopub.status.busy": "2026-05-25T10:34:57.182585Z",
     "iopub.status.idle": "2026-05-25T10:34:57.186279Z",
     "shell.execute_reply": "2026-05-25T10:34:57.185713Z"
    },
    "papermill": {
     "duration": 0.009697,
     "end_time": "2026-05-25T10:34:57.187565+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.177868+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# label_map = {\n",
    "#     \"Normal\": 0,\n",
    "#     \"Low\": 3,\n",
    "#     \"Mid\": 2,\n",
    "#     \"High\": 1,\n",
    "#     \"Very_Low\": 4\n",
    "# }\n",
    "# NUM_CLASSES = len(label_map)\n",
    "\n",
    "# ✅ MODIFIED: Class order — Normal → High → Mid → Low → Very_Low\n",
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
   "cell_type": "markdown",
   "id": "1d69e373",
   "metadata": {
    "papermill": {
     "duration": 0.00361,
     "end_time": "2026-05-25T10:34:57.194927+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.191317+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 2. GFCC Feature Extraction\n",
    "\n",
    "GFCC uses a **gammatone filterbank** (which models the human auditory system) instead of mel filterbanks used by MFCC. Steps:\n",
    "1. Frame the audio signal\n",
    "2. Apply gammatone filters (IIR, ERB-spaced center frequencies)\n",
    "3. Compute log energy in each filter channel\n",
    "4. Apply DCT to decorrelate → GFCC coefficients"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ff2bf9cc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:57.203899Z",
     "iopub.status.busy": "2026-05-25T10:34:57.203658Z",
     "iopub.status.idle": "2026-05-25T10:34:57.212218Z",
     "shell.execute_reply": "2026-05-25T10:34:57.211566Z"
    },
    "papermill": {
     "duration": 0.014707,
     "end_time": "2026-05-25T10:34:57.213523+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.198816+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def erb_space(low_freq, high_freq, n):\n",
    "    \"\"\"Generate ERB-spaced center frequencies for gammatone filterbank.\"\"\"\n",
    "    ear_q = 9.26449\n",
    "    min_bw = 24.7\n",
    "    cf_array = -(ear_q * min_bw) + np.exp(\n",
    "        np.arange(1, n + 1) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n\n",
    "    ) * (high_freq + ear_q * min_bw)\n",
    "    return cf_array\n",
    "\n",
    "\n",
    "def gammatone_filterbank(signal, sr, n_filters=64, low_freq=100.0):\n",
    "    \"\"\"\n",
    "    Apply a gammatone filterbank to a signal.\n",
    "    Returns: (n_filters, n_samples) array of filter outputs.\n",
    "    \"\"\"\n",
    "    high_freq = sr / 2.0\n",
    "    center_freqs = erb_space(low_freq, high_freq, n_filters)\n",
    "    output = np.zeros((n_filters, len(signal)))\n",
    "    for i, cf in enumerate(center_freqs):\n",
    "        b, a = gammatone(cf, 'iir', fs=sr)\n",
    "        from scipy.signal import lfilter\n",
    "        output[i] = np.abs(lfilter(b, a, signal))\n",
    "    return output\n",
    "\n",
    "\n",
    "def extract_gfcc(file_path):\n",
    "    \"\"\"\n",
    "    Extract GFCC features from an audio file.\n",
    "    Replaces extract_mfcc() — same interface, drop-in swap.\n",
    "    \"\"\"\n",
    "    y, sr = librosa.load(file_path, sr=SR)\n",
    "\n",
    "    frame_length = int(FRAME_LEN * sr)   # samples per frame\n",
    "    hop_length   = int(HOP_LEN  * sr)   # hop between frames\n",
    "\n",
    "    # 1. Apply gammatone filterbank to the full signal\n",
    "    gt_output = gammatone_filterbank(y, sr, n_filters=N_FILTERS, low_freq=LOW_FREQ)\n",
    "    # gt_output: (N_FILTERS, n_samples)\n",
    "\n",
    "    # 2. Frame the filter outputs and compute log energy per frame\n",
    "    n_frames = 1 + (gt_output.shape[1] - frame_length) // hop_length\n",
    "    frames = np.zeros((N_FILTERS, n_frames))\n",
    "    for t in range(n_frames):\n",
    "        start = t * hop_length\n",
    "        end   = start + frame_length\n",
    "        frame_energy = np.sum(gt_output[:, start:end] ** 2, axis=1)\n",
    "        frames[:, t] = np.log(frame_energy + 1e-10)\n",
    "\n",
    "    # 3. Apply DCT to get cepstral coefficients\n",
    "    gfcc = dct(frames, type=2, axis=0, norm='ortho')[:N_GFCC, :]\n",
    "    # gfcc: (N_GFCC, n_frames)\n",
    "\n",
    "    # 4. Fix length (pad or truncate to FIXED_LEN)\n",
    "    if gfcc.shape[1] < FIXED_LEN:\n",
    "        pad_width = FIXED_LEN - gfcc.shape[1]\n",
    "        gfcc = np.pad(gfcc, ((0, 0), (0, pad_width)), mode='constant')\n",
    "    else:\n",
    "        gfcc = gfcc[:, :FIXED_LEN]\n",
    "\n",
    "    return gfcc  # shape: (N_GFCC, FIXED_LEN)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fbf529a",
   "metadata": {
    "papermill": {
     "duration": 0.004358,
     "end_time": "2026-05-25T10:34:57.221649+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.217291+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 3. Precompute & Save GFCC Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6eda5410",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:57.230727Z",
     "iopub.status.busy": "2026-05-25T10:34:57.230262Z",
     "iopub.status.idle": "2026-05-25T10:34:57.236240Z",
     "shell.execute_reply": "2026-05-25T10:34:57.235496Z"
    },
    "papermill": {
     "duration": 0.012329,
     "end_time": "2026-05-25T10:34:57.237798+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.225469+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "SAVE_PATH = \"/kaggle/working/gfcc_features\"\n",
    "os.makedirs(SAVE_PATH, exist_ok=True)\n",
    "\n",
    "def save_gfcc_dataset(root_dir, split):\n",
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
    "            if not file.endswith(\".flac\"):\n",
    "                continue\n",
    "\n",
    "            file_path = os.path.join(sev_path, file)\n",
    "            gfcc = extract_gfcc(file_path)\n",
    "\n",
    "            save_file = os.path.join(save_sev_path, file.replace(\".flac\", \".npy\"))\n",
    "            np.save(save_file, gfcc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4ed80965",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:34:57.246989Z",
     "iopub.status.busy": "2026-05-25T10:34:57.246425Z",
     "iopub.status.idle": "2026-05-25T12:29:53.881612Z",
     "shell.execute_reply": "2026-05-25T12:29:53.880753Z"
    },
    "papermill": {
     "duration": 6896.641332,
     "end_time": "2026-05-25T12:29:53.883174+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:34:57.241842+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train-Very_Low: 100%|██████████| 6820/6820 [10:51<00:00, 10.47it/s]\n",
      "train-Mid: 100%|██████████| 6174/6174 [07:47<00:00, 13.19it/s]\n",
      "train-Low: 100%|██████████| 6200/6200 [08:34<00:00, 12.05it/s]\n",
      "train-Normal: 100%|██████████| 46410/46410 [39:58<00:00, 19.35it/s]\n",
      "train-High: 100%|██████████| 10850/10850 [10:57<00:00, 16.50it/s]\n",
      "test-Very_Low: 100%|██████████| 3086/3086 [04:28<00:00, 11.48it/s]\n",
      "test-Mid: 100%|██████████| 3100/3100 [03:40<00:00, 14.04it/s]\n",
      "test-Low: 100%|██████████| 3100/3100 [03:41<00:00, 14.02it/s]\n",
      "test-Normal: 100%|██████████| 23205/23205 [19:25<00:00, 19.90it/s]\n",
      "test-High: 100%|██████████| 5425/5425 [05:27<00:00, 16.55it/s]\n"
     ]
    }
   ],
   "source": [
    "save_gfcc_dataset(ROOT_PATH, \"train\")\n",
    "save_gfcc_dataset(ROOT_PATH, \"test\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed0e1371",
   "metadata": {
    "papermill": {
     "duration": 2.04169,
     "end_time": "2026-05-25T12:29:57.767017+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:29:55.725327+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 4. PyTorch Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bd7c5a76",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:01.536616Z",
     "iopub.status.busy": "2026-05-25T12:30:01.536273Z",
     "iopub.status.idle": "2026-05-25T12:30:01.543379Z",
     "shell.execute_reply": "2026-05-25T12:30:01.542503Z"
    },
    "papermill": {
     "duration": 1.847625,
     "end_time": "2026-05-25T12:30:01.545163+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:29:59.697538+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class GFCCDataset(Dataset):\n",
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
    "        gfcc = np.load(self.files[idx])  # shape: (N_GFCC, FIXED_LEN)\n",
    "        gfcc  = torch.tensor(gfcc, dtype=torch.float32).unsqueeze(0)  # (1, N_GFCC, FIXED_LEN)\n",
    "        label = torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "        return gfcc, label\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8cf3ce5e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:05.443475Z",
     "iopub.status.busy": "2026-05-25T12:30:05.442811Z",
     "iopub.status.idle": "2026-05-25T12:30:05.601315Z",
     "shell.execute_reply": "2026-05-25T12:30:05.600694Z"
    },
    "papermill": {
     "duration": 2.012396,
     "end_time": "2026-05-25T12:30:05.602914+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:03.590518+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_dataset = GFCCDataset(\"/kaggle/working/gfcc_features/train\")\n",
    "test_dataset  = GFCCDataset(\"/kaggle/working/gfcc_features/test\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ebaebca8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:09.521959Z",
     "iopub.status.busy": "2026-05-25T12:30:09.521554Z",
     "iopub.status.idle": "2026-05-25T12:30:09.581760Z",
     "shell.execute_reply": "2026-05-25T12:30:09.581144Z"
    },
    "papermill": {
     "duration": 1.9789,
     "end_time": "2026-05-25T12:30:09.583427+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:07.604527+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_size   = int(0.1 * len(train_dataset))\n",
    "train_size = len(train_dataset) - val_size\n",
    "\n",
    "train_ds, val_ds = random_split(train_dataset, [train_size, val_size])\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "458cd107",
   "metadata": {
    "papermill": {
     "duration": 2.015385,
     "end_time": "2026-05-25T12:30:13.413170+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:11.397785+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 5. CNN Model\n",
    "\n",
    "Architecture is unchanged — GFCC produces the same `(1, N_GFCC, FIXED_LEN)` shaped tensor as MFCC did."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "72111ef5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:17.265966Z",
     "iopub.status.busy": "2026-05-25T12:30:17.265212Z",
     "iopub.status.idle": "2026-05-25T12:30:17.271965Z",
     "shell.execute_reply": "2026-05-25T12:30:17.271290Z"
    },
    "papermill": {
     "duration": 2.028951,
     "end_time": "2026-05-25T12:30:17.273478+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:15.244527+00:00",
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
    "        # Adaptive pooling — avoids manual dimension calc\n",
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
    "        return x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "efba5cb9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:21.050638Z",
     "iopub.status.busy": "2026-05-25T12:30:21.049776Z",
     "iopub.status.idle": "2026-05-25T12:30:28.478972Z",
     "shell.execute_reply": "2026-05-25T12:30:28.478303Z"
    },
    "papermill": {
     "duration": 9.269268,
     "end_time": "2026-05-25T12:30:28.480825+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:19.211557+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model     = CNNModel().to(device)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=LR)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "448fbfa2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:32.352059Z",
     "iopub.status.busy": "2026-05-25T12:30:32.350992Z",
     "iopub.status.idle": "2026-05-25T12:30:32.355016Z",
     "shell.execute_reply": "2026-05-25T12:30:32.354399Z"
    },
    "papermill": {
     "duration": 1.85827,
     "end_time": "2026-05-25T12:30:32.356485+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:30.498215+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2cb98f1a",
   "metadata": {
    "papermill": {
     "duration": 1.836345,
     "end_time": "2026-05-25T12:30:36.211729+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:34.375384+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 6. Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "95d93126",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:40.155587Z",
     "iopub.status.busy": "2026-05-25T12:30:40.154747Z",
     "iopub.status.idle": "2026-05-25T12:30:40.159081Z",
     "shell.execute_reply": "2026-05-25T12:30:40.158400Z"
    },
    "papermill": {
     "duration": 1.93511,
     "end_time": "2026-05-25T12:30:40.160651+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:38.225541+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# !pip uninstall -y torch torchvision torchaudio\n",
    "# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "cf6c0aac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:44.000376Z",
     "iopub.status.busy": "2026-05-25T12:30:43.999396Z",
     "iopub.status.idle": "2026-05-25T12:30:44.004830Z",
     "shell.execute_reply": "2026-05-25T12:30:44.003970Z"
    },
    "papermill": {
     "duration": 2.034168,
     "end_time": "2026-05-25T12:30:44.006241+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:41.972073+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.10.0+cu128\n",
      "12.8\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "print(torch.__version__)\n",
    "print(torch.version.cuda)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "9cb73e93",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T12:30:47.864274Z",
     "iopub.status.busy": "2026-05-25T12:30:47.863410Z",
     "iopub.status.idle": "2026-05-25T13:06:26.962536Z",
     "shell.execute_reply": "2026-05-25T13:06:26.961552Z"
    },
    "papermill": {
     "duration": 2143.726716,
     "end_time": "2026-05-25T13:06:29.584563+00:00",
     "exception": false,
     "start_time": "2026-05-25T12:30:45.857847+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:42<00:00, 21.07it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20: Loss=522.9632, Val Acc=0.9652\n",
      "✅ Best model updated! Val Acc: 0.9652\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:45<00:00, 20.40it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/20: Loss=168.7718, Val Acc=0.9806\n",
      "✅ Best model updated! Val Acc: 0.9806\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.52it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/20: Loss=112.9726, Val Acc=0.9860\n",
      "✅ Best model updated! Val Acc: 0.9860\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.54it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/20: Loss=82.6683, Val Acc=0.9865\n",
      "✅ Best model updated! Val Acc: 0.9865\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:45<00:00, 20.48it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/20: Loss=66.8917, Val Acc=0.9876\n",
      "✅ Best model updated! Val Acc: 0.9876\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/20: Loss=56.9470, Val Acc=0.9888\n",
      "✅ Best model updated! Val Acc: 0.9888\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:45<00:00, 20.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/20: Loss=47.6941, Val Acc=0.9867\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.49it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/20: Loss=47.8007, Val Acc=0.9907\n",
      "✅ Best model updated! Val Acc: 0.9907\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.53it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/20: Loss=35.4647, Val Acc=0.9872\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.57it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/20: Loss=40.0184, Val Acc=0.9906\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.61it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/20: Loss=37.3720, Val Acc=0.9859\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.59it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/20: Loss=32.3622, Val Acc=0.9924\n",
      "✅ Best model updated! Val Acc: 0.9924\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.59it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/20: Loss=34.2756, Val Acc=0.9925\n",
      "✅ Best model updated! Val Acc: 0.9925\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:45<00:00, 20.48it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/20: Loss=33.0422, Val Acc=0.9920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.52it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/20: Loss=26.2244, Val Acc=0.9929\n",
      "✅ Best model updated! Val Acc: 0.9929\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.59it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/20: Loss=28.3790, Val Acc=0.9946\n",
      "✅ Best model updated! Val Acc: 0.9946\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.59it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/20: Loss=29.9166, Val Acc=0.9933\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 18/20: Loss=35.0219, Val Acc=0.9920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 19/20: Loss=25.4003, Val Acc=0.9944\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:44<00:00, 20.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 20/20: Loss=22.9193, Val Acc=0.9888\n",
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
    "print(\"✅ Model copied to working directory!\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b48a2d75",
   "metadata": {
    "papermill": {
     "duration": 2.487242,
     "end_time": "2026-05-25T13:06:34.589654+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:32.102412+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 7. Inference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "67c835ec",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:06:39.691092Z",
     "iopub.status.busy": "2026-05-25T13:06:39.690157Z",
     "iopub.status.idle": "2026-05-25T13:06:39.696309Z",
     "shell.execute_reply": "2026-05-25T13:06:39.695465Z"
    },
    "papermill": {
     "duration": 2.578767,
     "end_time": "2026-05-25T13:06:39.697839+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:37.119072+00:00",
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
    "    gfcc = extract_gfcc(file_path)                              # (N_GFCC, FIXED_LEN)\n",
    "    gfcc = torch.tensor(gfcc).unsqueeze(0).unsqueeze(0)        # (1, 1, N_GFCC, FIXED_LEN)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = model(gfcc)\n",
    "        pred   = torch.argmax(output, dim=1).item()\n",
    "\n",
    "    inv_map = {v: k for k, v in label_map.items()}\n",
    "    return inv_map[pred]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91de4e86",
   "metadata": {
    "papermill": {
     "duration": 2.515582,
     "end_time": "2026-05-25T13:06:44.815554+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:42.299972+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 8. Evaluation on Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1ee9d6e7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:06:49.767284Z",
     "iopub.status.busy": "2026-05-25T13:06:49.766999Z",
     "iopub.status.idle": "2026-05-25T13:06:49.771535Z",
     "shell.execute_reply": "2026-05-25T13:06:49.770715Z"
    },
    "papermill": {
     "duration": 2.481846,
     "end_time": "2026-05-25T13:06:49.773059+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:47.291213+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "96c5302b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:06:54.894153Z",
     "iopub.status.busy": "2026-05-25T13:06:54.893867Z",
     "iopub.status.idle": "2026-05-25T13:06:54.931840Z",
     "shell.execute_reply": "2026-05-25T13:06:54.931144Z"
    },
    "papermill": {
     "duration": 2.665176,
     "end_time": "2026-05-25T13:06:54.933328+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:52.268152+00:00",
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
       "    (4): Linear(in_features=64, out_features=5, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = CNNModel().to(device)\n",
    "model.load_state_dict(torch.load(\"best_model.pth\"))\n",
    "model.eval()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f203a56b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:06:59.868309Z",
     "iopub.status.busy": "2026-05-25T13:06:59.867533Z",
     "iopub.status.idle": "2026-05-25T13:07:13.536006Z",
     "shell.execute_reply": "2026-05-25T13:07:13.535322Z"
    },
    "papermill": {
     "duration": 16.135174,
     "end_time": "2026-05-25T13:07:13.537576+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:06:57.402402+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1185/1185 [00:13<00:00, 86.76it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Test Accuracy: 0.9620\n"
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
    "all_preds  = []\n",
    "all_labels = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for x, y in tqdm(test_loader):\n",
    "        x, y = x.to(device), y.to(device)\n",
    "\n",
    "        outputs = model(x)\n",
    "        preds   = torch.argmax(outputs, dim=1)\n",
    "\n",
    "        correct += (preds == y).sum().item()\n",
    "        total   += y.size(0)\n",
    "\n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(y.cpu().numpy())\n",
    "\n",
    "test_acc = correct / total\n",
    "print(f\"✅ Test Accuracy: {test_acc:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "fd62c595",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:07:18.597650Z",
     "iopub.status.busy": "2026-05-25T13:07:18.597109Z",
     "iopub.status.idle": "2026-05-25T13:07:20.479534Z",
     "shell.execute_reply": "2026-05-25T13:07:20.478791Z"
    },
    "papermill": {
     "duration": 4.41185,
     "end_time": "2026-05-25T13:07:20.481502+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:07:16.069652+00:00",
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
      "      Normal       0.99      1.00      1.00     23205\n",
      "        High       0.92      0.96      0.94      5425\n",
      "         Mid       0.92      0.81      0.86      3100\n",
      "         Low       0.95      0.88      0.91      3100\n",
      "    Very_Low       0.86      0.94      0.90      3086\n",
      "\n",
      "    accuracy                           0.96     37916\n",
      "   macro avg       0.93      0.92      0.92     37916\n",
      "weighted avg       0.96      0.96      0.96     37916\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "print(\"\\nClassification Report:\\n\")\n",
    "print(classification_report(all_labels, all_preds, target_names=label_map.keys()))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3ca009b7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:07:25.659413Z",
     "iopub.status.busy": "2026-05-25T13:07:25.658486Z",
     "iopub.status.idle": "2026-05-25T13:07:48.426236Z",
     "shell.execute_reply": "2026-05-25T13:07:48.425154Z"
    },
    "papermill": {
     "duration": 25.289701,
     "end_time": "2026-05-25T13:07:48.428003+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:07:23.138302+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1185/1185 [00:13<00:00, 87.06it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Saved results.csv and results.xlsx\n",
      "             filename  score_Normal  score_High  score_Mid  score_Low  \\\n",
      "0  M07_B3_CW57_M4.npy    -11.660396  -15.353193  -5.826336  12.632891   \n",
      "1  F02_B3_CW59_M3.npy    -49.274479  -37.898529 -13.961781   5.468629   \n",
      "2    F02_B3_D2_M6.npy    -43.362591  -29.229151 -10.274899  10.949599   \n",
      "3  M07_B3_CW50_M2.npy    -17.723063  -14.885586 -15.266958   6.584087   \n",
      "4  M07_B3_CW31_M8.npy    -19.951279  -16.563951  -6.707154  14.502682   \n",
      "\n",
      "   score_Very_Low   prob_Normal     prob_High      prob_Mid  prob_Low  \\\n",
      "0      -12.215232  2.815530e-11  7.011294e-13  9.621876e-09  1.000000   \n",
      "1       -8.354601  1.680236e-24  1.465152e-19  3.643178e-09  0.999999   \n",
      "2      -12.175982  2.585325e-24  3.552965e-18  6.057833e-10  1.000000   \n",
      "3      -20.415379  2.776764e-11  4.740667e-10  3.237512e-10  1.000000   \n",
      "4      -17.355858  1.088517e-15  3.220569e-14  6.147318e-10  1.000000   \n",
      "\n",
      "   prob_Very_Low predict class actual class  \n",
      "0   1.616583e-11           Low          Low  \n",
      "1   9.923095e-07           Low          Low  \n",
      "2   9.050828e-11           Low          Low  \n",
      "3   1.880533e-12           Low          Low  \n",
      "4   1.458855e-14           Low          Low  \n"
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
    "class_order =[\"Normal\", \"High\", \"Mid\", \"Low\", \"Very_Low\"]\n",
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
   "execution_count": 22,
   "id": "377b5b96",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:07:53.494692Z",
     "iopub.status.busy": "2026-05-25T13:07:53.493582Z",
     "iopub.status.idle": "2026-05-25T13:07:54.770484Z",
     "shell.execute_reply": "2026-05-25T13:07:54.769640Z"
    },
    "papermill": {
     "duration": 3.816906,
     "end_time": "2026-05-25T13:07:54.773204+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:07:50.956298+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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M9CzVy+0aBEIOzs7WFlZwcvLC4MHD8GPPy3B//5ZX+b35aRJk3H06HE899zzaNs2GLa2trC2toZ/QAAeeWQ0evcxLD+2s7PDf/9tw/c//Ij+AwbA1dVV+hvQt18/LFy0GNu376xWzd/bZDIZli5djn/+2YCHH34Efv7+0v/NZs2bY8SIB/DVV/OwcqVhluDESZPw+utv4J6+feEfEACVSlWyI3NAAEaPfhTbd+wsU0v5yJHDuHbtGgCgTZsgziQkIjIiqIs03NqJiIiIiIgavK/nzZPKPbzyyowyHy4QGdcz/uyzL8zywQkRUWPBJCERERERETUK+fn5aN8uGMnJyXB2dkZ0TCwcHBwsHRbVEzk5OWjdKhA5OTnw8fHBhYtR1doQjIioseNyYyIiIiIiahTs7Ozw9tvvACips/jzkppvhEKNz89Llkg7cL/99jtMEBIRlcKZhERERERERERERE0cZxISERERERERERE1cUwSEhERERERERERNXFMEhIRERERERERETVxTBISERERERERERE1cUwSEhERERERERERNXFMEhIRERERERERETVxTBISERERERERERE1cUwSEhERERERERERNXFMEhIRERERERERETVxTBISERERERERERE1cUwSEhERERERERERNXFMEtaCKIrQ6XQQRdHSoRAREREREREREdUYk4S1oNfrEXnxPPR6vaVDISIiIiIiIiIiqjEmCYmIiIiIiIiIiJo4JgmJiIiIiIiIiIiaOCYJiYiIiIiIiIiImjgmCYmIiIiIiIiIiJo4JgmJiIiIiIiIiIiaOIWlAyAiIiIiIiIiIssRRRFarRY6nQ6AaOlwqFoEyOVyKBQKCIJQq5GYJCQiIiIiIiIiaqKKi4uRfOM6CgryLR0K1YKtrR28fXyhVCprPAaThERERERERERETZBer0fc1StQKBRo1qwZlEplrWej0d0limJJojc5GXFXr6B1m7aQyWpWXZBJQiIiIiIiIiKiJqi4uBh6vR4BAQGws7OzdDhUQ7a2trCyssKVK1dQXFwMlUpVo3G4cQkRERERERERUZNUUn+wpjPPqP4wfA9rXlOSPwVERERERERERERNHJOERERERERERERETRyThEREREREREREZDFxcXGQyWQ4ffq0WftWV2BgIObPn2/2cRsKJgmJiIiIiIiIiKhOTJo0CTKZDDKZDEqlEi1btsSsWbOgVqulPgEBAbh+/TpCQ0PrLI6cnBy88847CAkJgY2NDXx8fHDvvffi77//hijWvI5fY8LdjYmIiIiIiIiIqM4MHz4cS5cuhUajwYkTJzBx4kQIgoDPP/8cACCXy+Ht7V1n98/Ozkbfvn1x8+ZNzJ07F926dYNCoUBERATefPNNDBo0CM7OznV2/4aCMwmJiIiIiIiIiKjOWFtbw9vbGwEBARg1ahSGDBmCHTt2SOdLLyHOysrCuHHj4OnpCVtbWwQFBWHZsmXljq3T6TB58mSEhIQgISGh3D5vv/024uLicPjwYUyYMAHt2rVDUFAQnn32WZw6dQr29vblXvf111+jY8eOsLe3R7NmzfDSSy8hLy9POh8fH4+HHnoIrq6usLe3R2hoKP79999qP4b6gjMJiYiIiIiIiIjorjh//jwOHjyI5s2bV9jnvffew8WLF/Hvv//C3d0dMTExKCwsLNOvqKgITz75JOLi4rB37154eHiU6aPX6/Hnn3/iySefhK+vb5nzFSUIAUAmk+Hbb79FYGAgYmNjMXXqVMyaNQuLFy8GAEybNg3FxcWIiIiAnZ0dLl68KI1X1cdQnzBJSEREREREREREdWbTpk1wcHCAVqtFUVERZDIZFixYUGH/xMREdO7cGeHh4QCAFi1alOmTl5eHBx54AEVFRdi1axecnJzKHSs9PR1ZWVkIDg6udtwzZsyQ/t2iRQvMnTsXL774opQkTEhIwCOPPIIOHToAAFq2bFmtx1DfMElIRERERERERER1ZuDAgVi8eDHy8/Mxf/58yOVyjB49usL+L7zwAh599FGcOnUK9957L0aNGoXevXub9HnyySfh7++PnTt3wsbGpsKxarMpyY4dO/DZZ58hKioKOTk50Gq1UKvVKCgogK2tLaZPn46XXnoJ27dvx+DBgzF69Gh07Nixyo+hvmFNQiIiIiKiUrILNIiITMfKvQmYtykGH/19CZ9viMb3269iy+kUxKbkcydEIiKiKrKzs0Pr1q3RqVMn/PLLLzh69Ch++eWXCvvfd999iIuLw4wZM3Djxg0MGTIEr7/+epk+Z8+exaFDhyq9t4eHB5ydnREVFVWtmOPi4vDggw+iQ4cOWLt2LY4fP46FCxcCAIqLiwEAU6ZMwZUrVzBu3DicP38e3bp1k2ZIVuUx1DdMEhIRERER3XKzQIN1R65jwX+x2BuZgfj0QhQU6yACKNbqkZ5bjOOx2fj1QBKW7IrH5Rt5dxyTiIiIDGQyGd566y289957ldbo8/DwwIQJE7Bq1Sp88803WLJkicn5F198EZ9++ilGjhyJiIiISu83duxY/Pbbb7h+/XqZ83l5edBqtWWOnzhxAnq9HvPmzUPPnj0RFBRU7vUBAQF44YUXsG7dOsycORM///xzlR9DfcMkIRERERE1eaIo4viVLHy//SouXsut0jUpN4vw56FrWH/8BtQaXR1HSERE1HiMGTMGcrkcixYtKvf87NmzsX79esTExODChQvYvHkzQkJCyvSbPn065s6diwcffBD79++v8H4ff/wxAgIC0LNnT6xcuRIXL15EdHQ0li5dirCwMJMdi29r3bo1NBoNFixYgNjYWKxatQo//vijSZ8ZM2Zg69atuHr1Kk6ePIk9e/ZIcVb1MdQnrElIRERERE2aVqfHltOpOB1/s8w5B5UCPs7WsFcpoNGJSMspQkpOEYxXGp9NyEFSZiGe7OMPFzvlXYyciIioYVIoFJg6dSq+/PJLvPjii2XOK5VKvP3224iLi4ONjQ369u2L33//vdyxZsyYAb1ejxEjRmDLli3l1v1zdXXFoUOH8Nlnn+Hjjz9GfHw8XFxc0KFDB3zxxRflbnrSqVMnzJs3D1988QXefvtt9OvXD5988gkmTJgg9dHpdJg2bRqSkpLg6OiI4cOH4+uvv672Y6gvBHWRhsVUakin0yHy4nmEtAuFXC63dDhEREREVE1anR5/Hb6OKyn5JsfdHZTo2doFXk7WEATB5FyeWouzCTmIvJYL4xfS9tZyPHmPP7ycVHchciIiotpTqwsRdzUWbdq0qXTzD6r/CgsLER0djRaBLaFS1ex7yeXGRERERNQk6fQi1h4xTRAKALq3csbIrt7wdlaVSRACgL1Kgd5BrnggzAsOKsMHxXlFOvy6PwmZecV3I3wiIiIis2KSkIiIiIiaHFEU8e+pFEQnGxKECrmAYR090bGZU7nJwdK8nFR4qKsP3OwNS4zzi3T4/UAS8ovKFkAnIiIiqs+YJCQiIiKiJud4bLZJDUK5TMCwDh7wd6ve8hwbpRwjunjBw8GQKMzML9khWa9nVR8iIiJqOJgkJCIiIqImJTGjENvOpkptAcC9oR7wcalZ/R6lQoahHT3haGPYEzA+vRB7ozJqGyoRERHRXcMkIRERERE1GUUaPdYfvwHjSX7dW7lUewZhaTZKOYZ19ISV3LBMeV9UBq6m5ldyFREREVH9wSQhERERETUZ28+lIitfI7VbetgiNMDBLGM72Vqhb7CbybGNJ5NRrNWbZXwiIiKiusQkIRERERE1CbGp+TgVZ6hDaKuU455gtyptUlJVLT3tEOxrL7VvFmix+0K62cYnIiIiqitMEhIRERFRo6fV6bHldIrJsQEhblAqzP9yuHsrF9hZy6X20StZuJZZaPb7EBEREZkTk4RERERE1Ogdjs5CZp5hmXFbHzv4utauDmFFlAoZ7mlruux465lUiCJ3OyYiIqL6i0lCIiIiImrUcgo02HfJsNOwtUKG7q1c6vSeAW42aOlpK7WvZalxPjG3Tu9JRERE9duePXsgk8mQnZ1t6VDKxSQhERERETVqEZEZ0OoMs/i6t3aGtZW8kivMo3srF8hlhnqHO8+nQcNNTIiIiMxi0qRJkMlk+Oyzz0yO//PPP5DJmO6qCT5rRERERNRopeUU4Uy8YbMSV3srBHnbV3KF+dirFOjYzFFq56q1OBabfVfuTURE1BSoVCp88cUXyMrKMtuYxcXFZhuroWGSkIiIiIgarV0X0mFcCbBHKxez7mZ8J52aOcJWaZi1ePByBoo0nE1IRET1k7pYh/i0Aot9qYt11Yp3yJAh8Pb2xqefflphn3Xr1iE0NBQqlQqBgYGYN2+eyfnAwEDMnTsXEyZMgJOTE55//nksX74cLi4u2LRpE4KDg2FnZ4cxY8agoKAAK1asQGBgIFxdXfHyyy9DpzPEvGrVKnTr1g2Ojo7w8fHBU089hdTU1Op9EyxIYekAiIiIiIjqwo0sNS7fyJPavi4q+NXRZiUVUchl6NTcEYeiS2Y4FBbrcfRKFvoGu93hSiIiorsv5WYRft4VZ7H7TxnUAs09bO/c8Ra5XI6PP/4YTz31FF5++WX4+/ubnD9x4gTGjh2LOXPmYOzYsTh48CCmTp0KNzc3TJw4Ueo3b948vPfee5g9ezYAYN++fSgoKMCCBQvw+++/Izc3F6NHj8YjjzwCZ2dnbN68GbGxsXj00UfRp08fjB07FgCg0Wjw4Ycfom3btkhNTcVrr72GSZMmYfPmzbV/cu4CJgmJiIiIqFHaF5Vh0q7rzUoqEuzrgLMJOcgvKplpcOhyJrq3ujt1EYmIiBq7hx9+GJ07d8acOXPwyy+/mJz75ptvMHjwYLz33nsAgKCgIFy8eBFfffWVSZJw0KBBeO2116T2vn37oNFosHjxYrRq1QoAMHr0aKxevRrJycmwt7dHu3btMHDgQOzevVtKEk6ePFkao2XLlvj222/RvXt35OXlwd7+7pQ7qQ0uNyYiIiKiRiflphqXjGYR+ruq4O6gtEgscpmALi2cpHaRVo8TV29WcgURERFVx2effYaVK1ciMjLS5HhkZCR69+5tcqxPnz6Ijo42WSbctWvXMmPa2tpKCUIA8PLyQosWLUySfV5eXkhLS5PaJ06cwEMPPYTmzZvD0dERAwYMAAAkJCTU6vHdLZxJSERERESNzv6oTJN210BnywRySxtve5yMu4mCW7MJj8SUzCZUyPmZPRER1R9eTtaYMqiFRe9fE/369cOwYcPw9ttvY8KECdW+3s7OrswxKysrk7YgCOUe0+tLag3n5+dj+PDhGDZsGFavXg0PDw8kJCRg+PDhDWYzFCYJiYiIiKhRycovRuS1XKnt66KCh2PN3nSYi1wmoIO/I45cKalNmKfW4VxCDrpYOHlJRERkTKWUV6smYH3y6aefokuXLggKCpKOhYSE4ODBgyb9Dhw4gKCgIMjl5i37ERUVhYyMDHz66acICAgAABw/ftys96hr/OiSiIiIiBqVozHZJjsad2nuVGHfu6mtrz2sFYaX3wejMyGKYiVXEBERUVV16NABTz31FBYsWCAdmzlzJnbu3Im5c+fi8uXLWLFiBRYtWmRSf9BcmjVrBqVSiQULFiA2NhYbNmzARx99ZPb71CUmCYmIiIio0VAX63A6Pltqu9pZwdvZsrMIb1MqZAjxc5DamXkaxKYWWDAiIiKixuWDDz6Qlv8CQFhYGP7880/8+eef6NChA+bMmYMPPvjAZNMSc/Hw8MCyZcuwdu1atG/fHp9//jm+/PJLs9+nLgnqIg0/vqwhnU6HyIvnEdIu1OzTVImIiIio+g5ezsTO84YC4gNC3NDau/7sJlhQpMXvh67h9gTC1l52eKKPv2WDIiKiJkutLkTc1Vi0adMGNjY2lg6HaqGwsBDR0dFoEdgSKlXNvpecSUhEREREjYJeFHEiNltq2yrlaOlZthC5JdlaK9DS01DrKSYlHxl5DaOYORERETVuTBISERERUaNwJTkf2QUaqR3iZw+ZTLBgROVr7+do0j5+JdsygRAREREZYZKQiIiIiBqF40azCGUCEOzrUHFnC/J0soaHg1Jqn024CY1OX8kVRERERHWPSUIiIiIiavCy8osRk5IvtVt42MJGWX9rRhtvYKLW6BF5LdeC0RARERExSUhEREREjcCpqzdN2u39HSvoWT8EetrCSm5YCn2yVPxEREREdxuThERERETUoOn1Is4kGJJszrZW8HRUVnKF5VnJZWjtbdhUJTGjEOm5RRaMiIiIiJo6JgmJiIiIqEGLSclHnlontYN97SEI9W/DktKCfUxrJp6O42xCIiIispwGkyT84fvFCApqDSdHe/S9pzeOHTtaaf9169aiY4dQODnao2tYZ/y3ZYvJeZW1VblfX8+bV5cPg4iIiIjM7JRRck0mwGSGXn3m5qCEu9EGJucSc6AXRQtGRERERE1Zg0gSrlnzF2bNegPvvPMuDh85ig4dOuLBB0YgNTW13P6HDh3E+KfHYeLESThy5BgefGgkxowZjQsXzkt94uITTb5+/GkJBEHAqIcfvlsPi4iIiIhqKU+tRXRyntRu7m4LlVX93bCkNOOEZp5ah/i0AgtGQ0RERE1Zg0gSfvftfEye/AwmTJiIkJB2WLhoMWxtbbFixfJy+y9auBBDhw7DzNdeQ3BICN5//wN06dIF3y9eLPXx9vY2+dq0cSP69x+Ali1bVhhHUVERcnJypK/cXO5CR0RERGRJF5JyYDz5LtjX3nLB1EBLTzsYr4w+l5hjuWCIiIgameXLl8PFxaVa10yaNAkPN9EJZPU+SVhcXIyTJ09i0KDB0jGZTIaBgwbhyOHD5V5z+MhhDBo0yOTYkHuH4siR8vunpKRgy5Z/MXHSpEpj+eKLz+Hp4SZ9tWrZonoPhoiIiIjM6lyC4UNbW6Ucvi4qC0ZTfbZKOfyMYo68lgeNTm/BiIiIiBqGipJ5e/bsgUwmQ3Z2NsaOHYtLly5ZILqGqd4nCdPT06HT6eDp5Wly3MvTCykpyeVek5KcDE8vr1L9PZGSklJu/9WrVsHBwQGjRlWeKZ41602kpmVIX1di46r+QIiIiIjIrDJyi3EjWy21W3raNogNS0pr7W2Y/Vis1ePyjbxKehMREVFV2djYwNPT884dCQCgsHQA9cGKFcvx+ONPQKWq/JNna2trWFtbS22dTldJbyIiIiKqS+dLLc1t49Owlhrf1tzdBgq5AK2uZN30uYQctPd3tHBURETUFOmL1dDeLH+C1d2gcPKCTGm+VQHLly/Hq6++iqysLOnYRx99hAULFqCwsBCPPfYY3N3dsXXrVpw6dcrk2q+++gpff/01iouLMXbsWMyfPx9WVlZmi60+qvdJQnd3d8jlcqSmmG5SkpKaAi8v73Kv8fL2RmqpWYMpqanwKjW7EAD279+Py5cvYfWvv5ovaCIiIiKqU6IomiQJnWwVcLVrmC/creQytHC3RUxKPgDgSko+Coq0sLWu9y/ViYiokdHeTEH27l8sdn/ngc9A6dG8zsb/9ddf8cknn2DRokXo06cP/vjjD3z99dcIDAw06bd79254e3tj165diImJweOPP47OnTvj2WefrbPY6oN6v9xYqVQiLCwMu3fvko7p9Xrs2b0bPXr2LPeanj16Yvfu3SbHdu3cgR49yvZfvnwpwsLC0LFjJ/MGTkRERER15kaWGpn5Gqnd2suuQS41vs14l2O9CFxM4gZ5REREd7Jp0yY4ODiYfN1///0V9l+4cCEmT56MSZMmISgoCLNnz0aHDh3K9HNxccHChQsRHByMBx54ACNGjMCuXbvKGbFxqfdJQgB4+ZUZWLr0F6xatRJRkZGYPm0q8vPzMX78BADA5MkT8e6770j9p06bhm3btmL+N9/gUlQU5s79ECdOnMCLL71kMm5OTg7+XrcOkyZNvquPh4iIiIhq51yiaRKttZddBT0bBl9nFWyUcqnNXY6JiIjubODAgTh16pTJ15IlSyrsf+nSJXTv3t3kWLdu3cr0a9++PeRyw99lb29vpKamlunX2DSINQxjxjyG9LQ0fPjhB0hJTkanTp2wYeMmaflwYmIiZDJDvrNXr95YsXIV3p8zB7Nnv4vWrdtgzZp1aN8+1GTcv/76E6Io4rGxj9/Vx0NERERENacXRVxIMiTRPByUcLBpmEuNb5PJBLTytMX5WzMIkzLVyMovhoud0sKRERFRU6Jw8oLzwGcsev/qsLOzQ+vWrU2OJSUl1TqO0rUHBUGAXq+v9bj1XYNIEgLAiy9NxYsvTS333PbtO8scGz36UYwe/WilY06Z8iymTGnc68mJiIiIGpu41ALkFxk2kDNeqtuQtfa2k5KEABB1PQ+92rhaMCIiImpqZEpVndYEtLS2bdvi2LFjGD9+vHTs+PHjFoyofmkQy42JiIiIiG4zXoorCEArz8aRJHSzV8JeZVjaFHmNdQmJiIjMadq0aVi6dClWrFiB6OhofPTRRzh79myDrmtsTkwSEhEREVGDodXpcel6ntT2dVZBZVTLryETBAEtPGyl9rVMNXILtRaMiIiIqHF56qmn8H//939444030LVrV1y9ehUTJkyASqWydGj1gqAu0oiWDqKh0ul0iLx4HiHtQk0KWhIRERFR3YhJzsPvB69J7X7BbgjysbdgROaVnK3GplMpUvu+Tp4Ib+ViwYiIiKgxU6sLEXc1Fm3atIGNjY2lw7GIoUOHwtvbGytXrrR0KLVSWFiI6OhotAhsCZWqZt/LBlOTkIiIiIgoymgWoSAAzd0b1xsaLydr2CjlKCwuqbkYeT2XSUIiIiIzKSgowA8//IBhw4ZBLpfj999/x44dO7Bt2zZLh1YvcLkxERERETUIelE0WWrs7aSCtVXjWs0hCAJaGCU+49MLUVDEJcdERETmIAgCtmzZgv79+yM8PBybNm3C2rVrMWTIEEuHVi9wJiERERERNQiJ6YUoKDbsatzSs3HNIrythYctIm8lQ0URuHwjH51bOFk4KiIioobPxsYG27dvt3QY9RZnEhIRERFRgxB13XS33+bujWNX49J8nFWwVhhepnOXYyIiIrobmCQkIiIionpPFEWTeoSejkrYWjeupca3yWQCmhktOb6alo8ija6SK4iIiIhqj0lCIiIiIqr3bmSpkVNoqM0X6GFrwWjqXgujx6fTA9HJ+RaMhoiIiJoCJgmJiIiIqN6LNJpFCAAtPBrnUuPb/FxsoJALUjuKS46JiIiojjFJSERERET1WslSY0OSzNXOCg42jXv/PYVcQICbYcnxlZR86PSiBSMiIiKixo5JQiIiIiKq19Jyi5GZp5HaLTwb91Lj25obJQmLdSLi0wssGA0RERE1dkwSEhEREVG9VnqpbWAjX2p8m7+bDQSjdgzrEhIREdXIwIEDMWPGjEr7BAYGYv78+XclnvqKSUIiIiIiqteMN+1wtFHAxc7KgtHcPSorOTwcraX25Rt5lfQmIiJqWiZNmgSZTIYXXnihzLmpU6dCJpNh0qRJAIB169Zh7ty5dzvEBodJQiIiIiKqt/LUWlzPUkvtZkZLcJsC48ebla9BRl6xBaMhIiKqXwICAvDnn3+isLBQOqZWq/H777+jWbNm0jFXV1c4ODhYIsQGpXFXfCYiIiKiBq30Etvm7k2jHuFtAe42OH41W2rHJOfBrbWr5QIiIqJGrVBTjOt5WRa7v6+9C2yslFXuHxYWhitXruDvv//GU089BQD4+++/0axZMwQGBkr9Bg4ciE6dOknLiVNTUzFlyhTs2LED3t7enGV4C5OERERERFRvRScblthayQV4OVlX0rvxcbWzgp21HPlFOgDA5Rv56MEkIRER1ZHreVmYf2yLxe4/o9t9aOXiVa1rJk2ahOXLl0tJwmXLlmHixImIiIio9Jrr169j165dsLKywiuvvILU1NRaxd4YcLkxEREREdVLOr2I2FTDTEI/VxVkMqGSKxofQRAQYLTkOCG9AEUavQUjIiIiql/GjRuH/fv3Iz4+HvHx8Thw4ADGjRtXYf/Lly9jy5Yt+Omnn9CzZ0907doVP//8s8mS5aaKMwmJiIiIqF6KTy9AsVaU2k1tqfFtAW42iLpeMqNSLwKxqfkI8WNdJSIiIgDw8PDAiBEjsHz5coiiiBEjRsDd3b3C/pGRkVAoFOjatat0LDg4GM7Oznch2vqNSUIiIiIiqpdK1yMMcG1am5bc5ueiglwmQKcvSZjGJDNJSEREdcPX3gUzut1n0fvXxKRJkzB9+nQAwMKFC80ZUpPCJCERERER1TuiKOLyDUM9Qg9HJVRKuQUjshyFXAYfZ2skZZbs8hydnAdRFCEITWvpNRER1T0bK2W1awLWB8OHD0dxcTEEQcCwYcMq7RscHAytVosTJ06gW7duAIBLly4hOzv7LkRav7EmIRERERHVOxl5GmTla6R2c7emOYvwtmZuhqXW+UU63MgusmA0RERE9YtcLsfFixdx4cIFyOWVf6jYtm1bDB8+HC+88AKOHDmCEydO4Nlnn4WNTdN+rQEwSUhERERE9VCM0a7GABDQROsR3hZQKkla+vkhIiJq6hwdHeHo6FilvkuXLoWvry8GDBiA0aNH49lnn4Wnp2cdR1j/cbkxEREREdU70Ub1CG2VcrjaWVkwGstzsFHA2dYK2QUlsytjUwvQL8TCQREREVnQsmXLKj3/v//9T/r37t27Tc55e3tj48aNJseefvpp8wXXQHEmIRERERHVK2qNDgnpBVK7mbsN6+8B8HNVSf9OyixEkUZnwWiIiIiosWGSkIiIiIjqlaupBbi1kS8AoFkTr0d4m7/R7s6iCMSlFVTSm4iIiKh6mCQkIiIionrlSophqbFMAHxdVJX0bjq8na0hM5pQGZvKJCERERGZD5OERERERFRviKKIWKMkoZeTNRRyvmQFACu5DF5O1lLbOJlKREREVFt8xUVERERE9UZGXjFuFmqldoArlxobM15ynJWvQVZ+sQWjISIiosaESUIiIiIiqjeupJguofVnPUITxpuXAEBsCpccExERkXkwSUhERERE9YbxUmMbpQwudlYWjKb+cbNXQmVleAkfm8olx0RERGQeTBISERERUb2g1ekRl26YGefnYgNBECq5oukRBMFkI5erqQXQG28FTURERFRDTBISERERUb2QkFEIrc6Q8ArgUuNyGdclLNLqcT1LbcFoiIiIqLFgkpCIiIiI6oXYUrv1lq6/RyX8XEyflytcckxERERmwCQhEREREdULV4yShG72VlBZyS0YTf1lp1LA2dZQq7F0cpWIiKgpmDRpEh5++GFLh9GoMElIRERERBaXW6hFak6x1OauxpXzN5pleS1LDXWxzoLREBERUWOgsHQARERERESld+kNcGWSsDJ+rjY4n5QLABBFID69EG197S0cFRERNXR6dRGKU9Isdn+llwdkKutajxMREYFZs2bhzJkzcHV1xfjx4/HRRx9BoVBg06ZNePrpp5Geng65XI7Tp08jLCwMs2bNwmeffQYAmDJlCoqKirBq1apax9KQMElIRERERBZnvNTYSi7A07H2bxAaMy8nawhCSYIQAOLSCpgkJCKiWitOSUPqz39Y7P6eUx6Hqrl/rca4du0aRowYgQkTJmDFihWIiorCc889B5VKhffffx99+/ZFbm4uTp06hfDwcERERMDd3R0RERHSGHv37sWsWbNq+3AaHC43JiIiIiKLEkURV1MLpLaPswoymWDBiOo/pUIGDwdDIvVqGusSEhERAcDixYsREBCAhQsXIjg4GKNGjcL777+Pr7/+Gnq9Hk5OTujcuTP27NkDoGTW4YwZM3Dq1Cnk5eXh2rVriImJQf/+/S37QCyASUIiIiIisqiUm0UoMKqp589djavE12iX47ScYuSrtRaMhoiIqH6IiopCr169IAiGDxz79OmDvLw8JCUlAQD69euHiIgIiKKIffv24ZFHHkFISAj279+PiIgI+Pr6ok2bNpZ6CBbD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I++55GI2bnwzXW/16jN56aUX0/ebm5t5++230XWdF194IV3+4gsvUFpaytx58wbV1lgsxgXnn8f//eY3fOQjH+U737mVvPx8rrryiqygzXEc3v8v7+MPv/89H//4J7jxxpuoPVrLlVf0HfZ1+sUv/xev18vqM8/kF7/8X37xy//lyiuv6rP+tOnTsW2b3/zm18c99lNPPcV5565h67atfP4LX+S2227nnHPeweOPP56u8/TTT/OeSy6mob6B66//Jldf8yXWrXuZd77zHPbt29fjmB/76EeJRaPcdNPNXH75FQDceut3uOLyT1NTM4vbb7+DL3zxataufYbzzl1Da2vrcdvZ3eHDhzl44ACLTl/UY9uNN97AjTf8N6eeehrf+c6tVFdXc8m7LyYaifR6rEWLF7Nu3csDboMQQgghhAw3FkIIIYQYQm1t7TQ2NhKPx3n11fXccsu38Hq9XHTxu9N1CgsL2b5jV9ZQ2cuvuIKFpy3gnnvu5qc/vReA1Weeyd13/5htW7cyd948Xn7pJTweD+effwEvvvgi//7ZzwHw4osvsGrV6kG3/ef33cu2bVv55f/+io9+9GMAXPVvn+H889Zwww3/zSc/9Wlyc3N55JG/sG7dOu6887t84YtXA/Bvn/l3Lr7owuM+xsc+9q988Qufp7q6mo997F+PW/+Tn/wUd/3oh1x15RXceccdnH322Zx51llceOFF5Ofnp+vZts0XPv85KiorWb/+NQoyQlmlVPr21//zaxQVFfHcP5+nqKgIgEsvvZQVy5dx88038vOf/zLr8U897TTuv/+B9P39+/dz8003csONN/G1r/1/6fL3ve99rFi+jJ/+9H+yyvtj+/btAFRVVWeVNzQ08L3v3slFF13Mnx96ON3L8Jvf/C9uv+3WXo9VXT2DxsZG6uvrKSsrG1A7hBBCCDGxSU9CIYQQQoghdPFF72LK5EpqZlbz0Y98mEAgyB//9BBTpkxJ1zEMIx0QOo5Dc3MzlmWxePES3nzjjXS9zqHKz7/wPJAKA5csXcq5557Liy+mehK2trayZcsWVq8efEj497//nYqKCj784Y+ky9xuN5/7/BcIh8M8/89/AvDkE0/gdru5/Ior0/V0XeffP/vZQbehu/Lycta/+jpXXfVvtLa2cO+9P+OTl32CqVMm8e1v35IOAN988w327dvLF7/wxayA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\n",
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
    "class_order = [\"Normal\", \"High\", \"Mid\", \"Low\", \"Very_Low\"]\n",
    "colors = {\n",
    "    \"Normal\":   \"#7ba7d4\",\n",
    "    \"High\":     \"#e8a87c\",\n",
    "    \"Mid\":      \"#6bb89e\",\n",
    "    \"Low\":      \"#e07b8a\",\n",
    "    \"Very_Low\": \"#9b8ec4\",\n",
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
    "ax.set_title(\"Class Score Distributions — GFCC-CNN (Real Scores)\",\n",
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
   "id": "034f7bb7",
   "metadata": {
    "papermill": {
     "duration": 2.506903,
     "end_time": "2026-05-25T13:07:59.932164+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:07:57.425261+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Single File Prediction (Optional)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e9ca0a12",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T13:08:04.979207Z",
     "iopub.status.busy": "2026-05-25T13:08:04.978905Z",
     "iopub.status.idle": "2026-05-25T13:08:04.983288Z",
     "shell.execute_reply": "2026-05-25T13:08:04.982314Z"
    },
    "papermill": {
     "duration": 2.544959,
     "end_time": "2026-05-25T13:08:04.985317+00:00",
     "exception": false,
     "start_time": "2026-05-25T13:08:02.440358+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# def predict(file_path, model, device):\n",
    "#     model.eval()\n",
    "#\n",
    "#     gfcc = extract_gfcc(file_path)\n",
    "#     gfcc = torch.tensor(gfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)\n",
    "#\n",
    "#     with torch.no_grad():\n",
    "#         output = model(gfcc)\n",
    "#         pred   = torch.argmax(output, dim=1).item()\n",
    "#\n",
    "#     inv_map = {v: k for k, v in label_map.items()}\n",
    "#     return inv_map[pred]\n",
    "#\n",
    "# file = \"/kaggle/input/YOUR_DATASET/test/Low/sample.wav\"\n",
    "# print(\"Prediction:\", predict(file, model, device))\n"
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
   "duration": 9206.768486,
   "end_time": "2026-05-25T13:08:10.549085+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-25T10:34:43.780599+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
