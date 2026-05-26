{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "54247381",
   "metadata": {
    "papermill": {
     "duration": 0.00424,
     "end_time": "2026-05-25T10:35:39.125904+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:39.121664+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# LFCC-CNN — Converted from your mfcc-cnn.ipynb\n",
    "\n",
    "Every cell below is your original code. Lines that changed are marked with `# ✅ CHANGED` and lines that are new are marked with `# ✅ NEW`. Nothing else is touched."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "122683a1",
   "metadata": {
    "papermill": {
     "duration": 0.003243,
     "end_time": "2026-05-25T10:35:39.132715+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:39.129472+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 1. Imports + Config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "74c31de5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:35:39.141708Z",
     "iopub.status.busy": "2026-05-25T10:35:39.141350Z",
     "iopub.status.idle": "2026-05-25T10:35:44.127693Z",
     "shell.execute_reply": "2026-05-25T10:35:44.126773Z"
    },
    "papermill": {
     "duration": 4.994384,
     "end_time": "2026-05-25T10:35:44.130325+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:39.135941+00:00",
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
    "from scipy.fftpack import dct   # ✅ NEW — needed for DCT step in LFCC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "24e3ef27",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:35:44.140797Z",
     "iopub.status.busy": "2026-05-25T10:35:44.140213Z",
     "iopub.status.idle": "2026-05-25T10:35:44.144753Z",
     "shell.execute_reply": "2026-05-25T10:35:44.144009Z"
    },
    "papermill": {
     "duration": 0.010372,
     "end_time": "2026-05-25T10:35:44.146212+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:44.135840+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ROOT_PATH = \"/kaggle/input/datasets/arthjs/dataset-ua-asr/dataset_UA_ASR\"  # change this\n",
    "\n",
    "SR = 16000\n",
    "N_LFCC = 20          # ✅ CHANGED: was N_MFCC = 20  — same number, just renamed\n",
    "N_FILTER = 40        # ✅ NEW: number of linear triangular filters in the filterbank\n",
    "FRAME_LEN = 0.025    # 25 ms — unchanged\n",
    "HOP_LEN = 0.010      # 10 ms — unchanged\n",
    "\n",
    "FIXED_LEN = 250   # unchanged\n",
    "\n",
    "BATCH_SIZE = 32      # unchanged\n",
    "EPOCHS = 20     # unchanged\n",
    "LR = 1e-3            # unchanged"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b590358e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:35:44.154957Z",
     "iopub.status.busy": "2026-05-25T10:35:44.154248Z",
     "iopub.status.idle": "2026-05-25T10:35:44.158157Z",
     "shell.execute_reply": "2026-05-25T10:35:44.157354Z"
    },
    "papermill": {
     "duration": 0.009759,
     "end_time": "2026-05-25T10:35:44.159737+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:44.149978+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# unchanged\n",
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
   "id": "99ea0f4e",
   "metadata": {
    "papermill": {
     "duration": 0.003314,
     "end_time": "2026-05-25T10:35:44.166641+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:44.163327+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 2. Feature Extraction\n",
    "\n",
    "### What changed and why\n",
    "\n",
    "Your original `extract_mfcc()` called `librosa.feature.mfcc()` which internally:\n",
    "1. Runs STFT\n",
    "2. Applies a **Mel filterbank** (filters bunched at low frequencies)\n",
    "3. Takes log\n",
    "4. Applies DCT → gives 20 MFCC coefficients\n",
    "\n",
    "The new `extract_lfcc()` does the **exact same 4 steps manually**, but in Step 2 uses a **Linear filterbank** (filters equally spaced across all frequencies). That single change is the entire difference between MFCC and LFCC.\n",
    "\n",
    "**Output shape is identical: `(20, 250)` — so the CNN does not change at all.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8d0a74a7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:35:44.175432Z",
     "iopub.status.busy": "2026-05-25T10:35:44.175112Z",
     "iopub.status.idle": "2026-05-25T10:35:44.185865Z",
     "shell.execute_reply": "2026-05-25T10:35:44.185325Z"
    },
    "papermill": {
     "duration": 0.016808,
     "end_time": "2026-05-25T10:35:44.187333+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:44.170525+00:00",
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
    "SAVE_PATH = \"/kaggle/working/lfcc_features\"   # ✅ CHANGED: was mfcc_features\n",
    "os.makedirs(SAVE_PATH, exist_ok=True)\n",
    "\n",
    "\n",
    "# ✅ NEW function — builds the linear triangular filterbank\n",
    "# This replaces what librosa does internally for Mel filters\n",
    "def build_linear_filterbank(n_filters, n_fft, sr):\n",
    "    n_bins = n_fft // 2 + 1                        # number of FFT frequency bins\n",
    "    freq_bins = np.linspace(0, sr / 2, n_bins)     # Hz value of each FFT bin\n",
    "\n",
    "    # ← THIS is the key line: equally spaced filter edges in Hz\n",
    "    # MFCC uses mel-spaced edges (bunched at low frequencies)\n",
    "    # LFCC uses linearly-spaced edges (equal spacing everywhere)\n",
    "    filter_points = np.linspace(0, sr / 2, n_filters + 2)\n",
    "\n",
    "    fbank = np.zeros((n_filters, n_bins))\n",
    "\n",
    "    for m in range(1, n_filters + 1):\n",
    "        f_left   = filter_points[m - 1]   # left edge of triangle\n",
    "        f_center = filter_points[m]       # peak of triangle\n",
    "        f_right  = filter_points[m + 1]  # right edge of triangle\n",
    "\n",
    "        for k, f in enumerate(freq_bins):\n",
    "            if f_left <= f <= f_center:\n",
    "                fbank[m - 1, k] = (f - f_left) / (f_center - f_left + 1e-10)\n",
    "            elif f_center < f <= f_right:\n",
    "                fbank[m - 1, k] = (f_right - f) / (f_right - f_center + 1e-10)\n",
    "\n",
    "    return fbank   # shape: (40, n_bins)\n",
    "\n",
    "\n",
    "# ✅ CHANGED: was extract_mfcc() — now extract_lfcc()\n",
    "def extract_lfcc(file_path):\n",
    "    y, sr = librosa.load(file_path, sr=SR)   # same as before\n",
    "\n",
    "    n_fft      = int(FRAME_LEN * SR)         # 400 samples = 25ms window\n",
    "    hop_length = int(HOP_LEN * SR)           # 160 samples = 10ms hop\n",
    "\n",
    "    # Step 1: STFT → power spectrum  (same concept as inside librosa.feature.mfcc)\n",
    "    stft       = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)\n",
    "    power_spec = np.abs(stft) ** 2           # shape: (n_fft/2+1, T)\n",
    "\n",
    "    # Step 2: Apply LINEAR filterbank  (← only difference from MFCC)\n",
    "    fbank           = build_linear_filterbank(N_FILTER, n_fft, SR)\n",
    "    filter_energies = np.dot(fbank, power_spec)   # shape: (40, T)\n",
    "\n",
    "    # Step 3: Log compression  (same as inside librosa.feature.mfcc)\n",
    "    log_energies = np.log(filter_energies + 1e-10)\n",
    "\n",
    "    # Step 4: DCT → keep first N_LFCC coefficients  (same as inside librosa.feature.mfcc)\n",
    "    lfcc = dct(log_energies, type=2, axis=0, norm='ortho')[:N_LFCC, :]\n",
    "    # shape: (20, T) — identical to MFCC output shape\n",
    "\n",
    "    # Pad or truncate to FIXED_LEN — exactly same as your original code\n",
    "    if lfcc.shape[1] < FIXED_LEN:\n",
    "        pad_width = FIXED_LEN - lfcc.shape[1]\n",
    "        lfcc = np.pad(lfcc, ((0, 0), (0, pad_width)), mode='constant')\n",
    "    else:\n",
    "        lfcc = lfcc[:, :FIXED_LEN]\n",
    "\n",
    "    return lfcc   # shape: (20, 250) — same as extract_mfcc() returned\n",
    "\n",
    "\n",
    "# ✅ CHANGED: was save_mfcc_dataset — now save_lfcc_dataset\n",
    "def save_lfcc_dataset(root_dir, split):\n",
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
    "\n",
    "            lfcc = extract_lfcc(file_path)   # ✅ CHANGED: was extract_mfcc\n",
    "\n",
    "            save_file = os.path.join(save_sev_path, file.replace(\".flac\", \".npy\"))\n",
    "            np.save(save_file, lfcc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e4683073",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T10:35:44.194938Z",
     "iopub.status.busy": "2026-05-25T10:35:44.194724Z",
     "iopub.status.idle": "2026-05-25T11:04:58.023804Z",
     "shell.execute_reply": "2026-05-25T11:04:58.023181Z"
    },
    "papermill": {
     "duration": 1753.835288,
     "end_time": "2026-05-25T11:04:58.025962+00:00",
     "exception": false,
     "start_time": "2026-05-25T10:35:44.190674+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train-Very_Low: 100%|██████████| 6820/6820 [01:58<00:00, 57.58it/s]\n",
      "train-Mid: 100%|██████████| 6174/6174 [01:36<00:00, 64.31it/s]\n",
      "train-Low: 100%|██████████| 6200/6200 [01:39<00:00, 62.20it/s]\n",
      "train-Normal: 100%|██████████| 46410/46410 [10:57<00:00, 70.56it/s]\n",
      "train-High: 100%|██████████| 10850/10850 [02:56<00:00, 61.60it/s]\n",
      "test-Very_Low: 100%|██████████| 3086/3086 [00:52<00:00, 59.24it/s]\n",
      "test-Mid: 100%|██████████| 3100/3100 [00:48<00:00, 63.81it/s]\n",
      "test-Low: 100%|██████████| 3100/3100 [00:46<00:00, 66.37it/s]\n",
      "test-Normal: 100%|██████████| 23205/23205 [05:34<00:00, 69.43it/s]\n",
      "test-High: 100%|██████████| 5425/5425 [02:02<00:00, 44.28it/s]\n"
     ]
    }
   ],
   "source": [
    "save_lfcc_dataset(ROOT_PATH, \"train\")   # ✅ CHANGED: was save_mfcc_dataset\n",
    "save_lfcc_dataset(ROOT_PATH, \"test\")    # ✅ CHANGED: was save_mfcc_dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "341e5f2c",
   "metadata": {
    "papermill": {
     "duration": 0.650295,
     "end_time": "2026-05-25T11:04:59.299345+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:04:58.649050+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 3. Dataset Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "665487e8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:00.372823Z",
     "iopub.status.busy": "2026-05-25T11:05:00.372096Z",
     "iopub.status.idle": "2026-05-25T11:05:00.379230Z",
     "shell.execute_reply": "2026-05-25T11:05:00.378509Z"
    },
    "papermill": {
     "duration": 0.541328,
     "end_time": "2026-05-25T11:05:00.380882+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:04:59.839554+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "# ✅ CHANGED: class name was MFCCDataset — now LFCCDataset\n",
    "# Everything inside is identical — loads .npy files, returns (tensor, label)\n",
    "class LFCCDataset(Dataset):\n",
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
    "        lfcc = np.load(self.files[idx])   # loads (20, 250) array\n",
    "\n",
    "        lfcc  = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0)  # → (1, 20, 250)\n",
    "        label = torch.tensor(self.labels[idx], dtype=torch.long)\n",
    "\n",
    "        return lfcc, label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "280b3ade",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:01.630665Z",
     "iopub.status.busy": "2026-05-25T11:05:01.629854Z",
     "iopub.status.idle": "2026-05-25T11:05:01.788710Z",
     "shell.execute_reply": "2026-05-25T11:05:01.788065Z"
    },
    "papermill": {
     "duration": 0.73991,
     "end_time": "2026-05-25T11:05:01.790508+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:01.050598+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# ✅ CHANGED: class name + folder path (mfcc_features → lfcc_features)\n",
    "train_dataset = LFCCDataset(\"/kaggle/working/lfcc_features/train\")\n",
    "test_dataset  = LFCCDataset(\"/kaggle/working/lfcc_features/test\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "5c02bb93",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:02.954307Z",
     "iopub.status.busy": "2026-05-25T11:05:02.953327Z",
     "iopub.status.idle": "2026-05-25T11:05:02.988278Z",
     "shell.execute_reply": "2026-05-25T11:05:02.987456Z"
    },
    "papermill": {
     "duration": 0.660621,
     "end_time": "2026-05-25T11:05:02.990054+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:02.329433+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# unchanged\n",
    "val_size   = int(0.1 * len(train_dataset))\n",
    "train_size = len(train_dataset) - val_size\n",
    "\n",
    "train_ds, val_ds = random_split(train_dataset, [train_size, val_size])\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0abdf138",
   "metadata": {
    "papermill": {
     "duration": 0.62706,
     "end_time": "2026-05-25T11:05:04.150656+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:03.523596+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 4. CNN Model — 100% Unchanged\n",
    "\n",
    "The CNN receives shape `(batch, 1, 20, 250)`. LFCC also produces `(20, 250)` — same shape as MFCC — so the model needs zero changes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "eabc0c29",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:05.216722Z",
     "iopub.status.busy": "2026-05-25T11:05:05.216401Z",
     "iopub.status.idle": "2026-05-25T11:05:05.222889Z",
     "shell.execute_reply": "2026-05-25T11:05:05.222272Z"
    },
    "papermill": {
     "duration": 0.537576,
     "end_time": "2026-05-25T11:05:05.224251+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:04.686675+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# unchanged\n",
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
   "execution_count": 10,
   "id": "d47e3318",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:06.385679Z",
     "iopub.status.busy": "2026-05-25T11:05:06.384947Z",
     "iopub.status.idle": "2026-05-25T11:05:12.552442Z",
     "shell.execute_reply": "2026-05-25T11:05:12.551702Z"
    },
    "papermill": {
     "duration": 6.717908,
     "end_time": "2026-05-25T11:05:12.554282+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:05.836374+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# unchanged\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model     = CNNModel().to(device)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "871ee87e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:13.746805Z",
     "iopub.status.busy": "2026-05-25T11:05:13.745923Z",
     "iopub.status.idle": "2026-05-25T11:05:18.634241Z",
     "shell.execute_reply": "2026-05-25T11:05:18.633326Z"
    },
    "papermill": {
     "duration": 5.434294,
     "end_time": "2026-05-25T11:05:18.636060+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:13.201766+00:00",
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
   "cell_type": "markdown",
   "id": "0d523055",
   "metadata": {
    "papermill": {
     "duration": 0.643373,
     "end_time": "2026-05-25T11:05:19.824781+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:19.181408+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 5. Training Loop — unchanged except model save filename"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "6ebbda87",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:05:20.901638Z",
     "iopub.status.busy": "2026-05-25T11:05:20.901214Z",
     "iopub.status.idle": "2026-05-25T11:38:47.438198Z",
     "shell.execute_reply": "2026-05-25T11:38:47.437214Z"
    },
    "papermill": {
     "duration": 2007.079081,
     "end_time": "2026-05-25T11:38:47.440092+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:05:20.361011+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:39<00:00, 21.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20: Loss=444.0518, Val Acc=0.9635\n",
      "✅ Best model updated! Val Acc: 0.9635\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.77it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/20: Loss=168.3194, Val Acc=0.9600\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.83it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/20: Loss=102.4565, Val Acc=0.9838\n",
      "✅ Best model updated! Val Acc: 0.9838\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.89it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/20: Loss=72.0404, Val Acc=0.9761\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.04it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/20: Loss=59.4426, Val Acc=0.9919\n",
      "✅ Best model updated! Val Acc: 0.9919\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.90it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/20: Loss=47.5968, Val Acc=0.9939\n",
      "✅ Best model updated! Val Acc: 0.9939\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.09it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/20: Loss=43.2033, Val Acc=0.9919\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/20: Loss=45.7515, Val Acc=0.9901\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.00it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/20: Loss=31.2665, Val Acc=0.9927\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.02it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/20: Loss=30.9368, Val Acc=0.9906\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.01it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/20: Loss=27.1064, Val Acc=0.9935\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:39<00:00, 21.72it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/20: Loss=37.2647, Val Acc=0.9928\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.94it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/20: Loss=36.9267, Val Acc=0.9911\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.03it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/20: Loss=24.5165, Val Acc=0.9920\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:38<00:00, 21.89it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/20: Loss=28.2126, Val Acc=0.9924\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.02it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/20: Loss=17.0678, Val Acc=0.9923\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.06it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/20: Loss=24.1742, Val Acc=0.9895\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.03it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 18/20: Loss=27.8022, Val Acc=0.9937\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.01it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 19/20: Loss=15.1675, Val Acc=0.9932\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 2151/2151 [01:37<00:00, 22.08it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 20/20: Loss=30.9145, Val Acc=0.9942\n",
      "✅ Best model updated! Val Acc: 0.9942\n",
      "✅ Model copied to working directory!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import shutil\n",
    "\n",
    "best_val_acc = 0\n",
    "save_path = \"/tmp/best_model_lfcc.pth\"   # ✅ CHANGED: filename reflects lfcc\n",
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
    "    total   = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in val_loader:\n",
    "            x, y = x.to(device), y.to(device)\n",
    "            outputs = model(x)\n",
    "            preds   = torch.argmax(outputs, dim=1)\n",
    "            correct += (preds == y).sum().item()\n",
    "            total   += y.size(0)\n",
    "\n",
    "    val_acc = correct / total\n",
    "    print(f\"Epoch {epoch+1}/{EPOCHS}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}\")\n",
    "\n",
    "    if val_acc > best_val_acc:\n",
    "        best_val_acc = val_acc\n",
    "        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)\n",
    "        print(f\"✅ Best model updated! Val Acc: {val_acc:.4f}\")\n",
    "\n",
    "shutil.copy(save_path, \"./best_model_lfcc.pth\")   # ✅ CHANGED: filename\n",
    "print(\"✅ Model copied to working directory!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "298887e3",
   "metadata": {
    "papermill": {
     "duration": 1.069368,
     "end_time": "2026-05-25T11:38:49.693591+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:48.624223+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 6. Predict Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1fb9a638",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:38:52.017312Z",
     "iopub.status.busy": "2026-05-25T11:38:52.016492Z",
     "iopub.status.idle": "2026-05-25T11:38:52.022131Z",
     "shell.execute_reply": "2026-05-25T11:38:52.021444Z"
    },
    "papermill": {
     "duration": 1.193332,
     "end_time": "2026-05-25T11:38:52.023573+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:50.830241+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def predict(file_path, model_path=\"best_model_lfcc.pth\"):   # ✅ CHANGED: filename\n",
    "    model = CNNModel()\n",
    "    model.load_state_dict(torch.load(model_path))\n",
    "    model.eval()\n",
    "\n",
    "    lfcc = extract_lfcc(file_path)                          # ✅ CHANGED: was extract_mfcc\n",
    "    lfcc = torch.tensor(lfcc).unsqueeze(0).unsqueeze(0)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = model(lfcc)\n",
    "        pred   = torch.argmax(output, dim=1).item()\n",
    "\n",
    "    inv_map = {v: k for k, v in label_map.items()}\n",
    "    return inv_map[pred]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8fec357b",
   "metadata": {
    "papermill": {
     "duration": 1.064828,
     "end_time": "2026-05-25T11:38:54.245032+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:53.180204+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 7. Test Evaluation — unchanged"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f1d605e5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:38:56.532819Z",
     "iopub.status.busy": "2026-05-25T11:38:56.532500Z",
     "iopub.status.idle": "2026-05-25T11:38:56.537209Z",
     "shell.execute_reply": "2026-05-25T11:38:56.536352Z"
    },
    "papermill": {
     "duration": 1.150709,
     "end_time": "2026-05-25T11:38:56.538781+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:55.388072+00:00",
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
   "execution_count": 15,
   "id": "c60b95e5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:38:58.762534Z",
     "iopub.status.busy": "2026-05-25T11:38:58.761713Z",
     "iopub.status.idle": "2026-05-25T11:38:58.793194Z",
     "shell.execute_reply": "2026-05-25T11:38:58.792364Z"
    },
    "papermill": {
     "duration": 1.089074,
     "end_time": "2026-05-25T11:38:58.794728+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:57.705654+00:00",
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
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = CNNModel().to(device)\n",
    "model.load_state_dict(torch.load(\"best_model_lfcc.pth\"))   # ✅ CHANGED: filename\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3eacb97c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:39:01.001520Z",
     "iopub.status.busy": "2026-05-25T11:39:01.000920Z",
     "iopub.status.idle": "2026-05-25T11:39:14.526090Z",
     "shell.execute_reply": "2026-05-25T11:39:14.525189Z"
    },
    "papermill": {
     "duration": 14.574005,
     "end_time": "2026-05-25T11:39:14.527697+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:38:59.953692+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1185/1185 [00:13<00:00, 87.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Test Accuracy: 0.9602\n"
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
    "# unchanged\n",
    "correct    = 0\n",
    "total      = 0\n",
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
    "print(f\"✅ Test Accuracy: {test_acc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "035cd6ba",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:39:16.813804Z",
     "iopub.status.busy": "2026-05-25T11:39:16.813396Z",
     "iopub.status.idle": "2026-05-25T11:39:17.951813Z",
     "shell.execute_reply": "2026-05-25T11:39:17.950800Z"
    },
    "papermill": {
     "duration": 2.294061,
     "end_time": "2026-05-25T11:39:17.953524+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:39:15.659463+00:00",
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
      "      Normal       1.00      1.00      1.00     23205\n",
      "        High       0.96      0.94      0.95      5425\n",
      "         Mid       0.82      0.83      0.82      3100\n",
      "         Low       0.90      0.90      0.90      3100\n",
      "    Very_Low       0.87      0.90      0.88      3086\n",
      "\n",
      "    accuracy                           0.96     37916\n",
      "   macro avg       0.91      0.91      0.91     37916\n",
      "weighted avg       0.96      0.96      0.96     37916\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# unchanged\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "print(\"\\nClassification Report:\\n\")\n",
    "print(classification_report(all_labels, all_preds, target_names=label_map.keys()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "69881951",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:39:20.222615Z",
     "iopub.status.busy": "2026-05-25T11:39:20.221583Z",
     "iopub.status.idle": "2026-05-25T11:39:41.822778Z",
     "shell.execute_reply": "2026-05-25T11:39:41.821757Z"
    },
    "papermill": {
     "duration": 22.673413,
     "end_time": "2026-05-25T11:39:41.824502+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:39:19.151089+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1185/1185 [00:13<00:00, 88.97it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Saved results.csv and results.xlsx\n",
      "             filename  score_Normal  score_High  score_Mid  score_Low  \\\n",
      "0    M05_B3_LL_M5.npy   -110.311195   -9.183969   7.451621 -19.686024   \n",
      "1  F04_B3_CW86_M7.npy    -90.293503   -8.601516  11.946863  -9.997938   \n",
      "2    M11_B3_LT_M3.npy   -135.580322  -28.213480  20.240873 -29.822094   \n",
      "3    F04_B3_LJ_M5.npy    -59.870068   -4.091112   7.243614  -2.144886   \n",
      "4    F04_B3_LE_M4.npy    -15.119472    1.315661  -2.405355  -3.338664   \n",
      "\n",
      "   score_Very_Low   prob_Normal     prob_High  prob_Mid      prob_Low  \\\n",
      "0       -9.125422  0.000000e+00  5.960123e-08  1.000000  1.637835e-12   \n",
      "1       -9.191220  4.203895e-45  1.191113e-09  1.000000  2.947772e-10   \n",
      "2      -25.312559  0.000000e+00  9.047773e-22  1.000000  1.811050e-22   \n",
      "3       -5.230834  7.126180e-30  1.194946e-05  0.999901  8.367262e-05   \n",
      "4       -1.750513  6.741499e-08  9.256441e-01  0.022409  8.812443e-03   \n",
      "\n",
      "   prob_Very_Low predict class actual class  \n",
      "0   6.319481e-08           Mid          Mid  \n",
      "1   6.604601e-10           Mid          Mid  \n",
      "2   1.645871e-20           Mid          Mid  \n",
      "3   3.822723e-06           Mid          Mid  \n",
      "4   4.313417e-02          High          Mid  \n"
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
   "cell_type": "markdown",
   "id": "3a7cc27c",
   "metadata": {
    "papermill": {
     "duration": 1.166167,
     "end_time": "2026-05-25T11:39:44.228744+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:39:43.062577+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## 8. Single File Prediction (unchanged logic)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9fc9cdca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T11:39:46.467712Z",
     "iopub.status.busy": "2026-05-25T11:39:46.466898Z",
     "iopub.status.idle": "2026-05-25T11:39:46.471174Z",
     "shell.execute_reply": "2026-05-25T11:39:46.470364Z"
    },
    "papermill": {
     "duration": 1.178953,
     "end_time": "2026-05-25T11:39:46.472762+00:00",
     "exception": false,
     "start_time": "2026-05-25T11:39:45.293809+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# def predict(file_path, model, device):\n",
    "#     model.eval()\n",
    "\n",
    "#     lfcc = extract_lfcc(file_path)                                        # ✅ CHANGED\n",
    "#     lfcc = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)\n",
    "\n",
    "#     with torch.no_grad():\n",
    "#         output = model(lfcc)\n",
    "#         pred = torch.argmax(output, dim=1).item()\n",
    "\n",
    "#     inv_map = {v:k for k,v in label_map.items()}\n",
    "#     return inv_map[pred]\n",
    "\n",
    "# file = \"/kaggle/input/YOUR_DATASET/test/Low/sample.flac\"\n",
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
   "duration": 3854.268816,
   "end_time": "2026-05-25T11:39:50.827485+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-25T10:35:36.558669+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
