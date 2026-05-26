{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "29592ca8",
   "metadata": {
    "papermill": {
     "duration": 0.005144,
     "end_time": "2026-05-25T07:32:44.732604+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:44.727460+00:00",
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
   "id": "faf379fe",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:44.742233Z",
     "iopub.status.busy": "2026-05-25T07:32:44.741910Z",
     "iopub.status.idle": "2026-05-25T07:32:54.964912Z",
     "shell.execute_reply": "2026-05-25T07:32:54.963964Z"
    },
    "papermill": {
     "duration": 10.230174,
     "end_time": "2026-05-25T07:32:54.966815+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:44.736641+00:00",
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
   "id": "9b24d6e2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:54.976555Z",
     "iopub.status.busy": "2026-05-25T07:32:54.975839Z",
     "iopub.status.idle": "2026-05-25T07:32:54.980756Z",
     "shell.execute_reply": "2026-05-25T07:32:54.979791Z"
    },
    "papermill": {
     "duration": 0.011187,
     "end_time": "2026-05-25T07:32:54.982225+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:54.971038+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "ROOT_PATH = \"/kaggle/input/datasets/artharking/severity-dau-kdah/Severity_DAU_KDAH\"  # change this\n",
    "\n",
    "SR = 16000\n",
    "N_GFCC = 20           # number of GFCC coefficients (same as N_MFCC was)\n",
    "N_FILTERS = 64        # number of gammatone filters\n",
    "FRAME_LEN = 0.025     # 25 ms\n",
    "HOP_LEN = 0.010       # 10 ms\n",
    "LOW_FREQ = 100.0      # lowest center frequency for gammatone bank\n",
    "\n",
    "FIXED_LEN = 398       # time frames (important for CNN)\n",
    "\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 20\n",
    "LR = 1e-3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fbb493af",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:54.991877Z",
     "iopub.status.busy": "2026-05-25T07:32:54.991484Z",
     "iopub.status.idle": "2026-05-25T07:32:54.995246Z",
     "shell.execute_reply": "2026-05-25T07:32:54.994480Z"
    },
    "papermill": {
     "duration": 0.009615,
     "end_time": "2026-05-25T07:32:54.996645+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:54.987030+00:00",
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
    "    \"N\":   0,\n",
    "    \"L\":     1,\n",
    "    \"M\":      2,\n",
    "    \"H\":      3\n",
    "}\n",
    "NUM_CLASSES = len(label_map)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a01d552d",
   "metadata": {
    "papermill": {
     "duration": 0.00354,
     "end_time": "2026-05-25T07:32:55.003864+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:55.000324+00:00",
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
   "id": "245f7764",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:55.012377Z",
     "iopub.status.busy": "2026-05-25T07:32:55.012040Z",
     "iopub.status.idle": "2026-05-25T07:32:55.020845Z",
     "shell.execute_reply": "2026-05-25T07:32:55.020234Z"
    },
    "papermill": {
     "duration": 0.014723,
     "end_time": "2026-05-25T07:32:55.022204+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:55.007481+00:00",
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
   "id": "f5794f35",
   "metadata": {
    "papermill": {
     "duration": 0.003914,
     "end_time": "2026-05-25T07:32:55.029837+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:55.025923+00:00",
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
   "id": "25df503d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:55.038428Z",
     "iopub.status.busy": "2026-05-25T07:32:55.038209Z",
     "iopub.status.idle": "2026-05-25T07:32:55.044069Z",
     "shell.execute_reply": "2026-05-25T07:32:55.043518Z"
    },
    "papermill": {
     "duration": 0.011937,
     "end_time": "2026-05-25T07:32:55.045506+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:55.033569+00:00",
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
    "            if not file.endswith(\".wav\"):\n",
    "                continue\n",
    "\n",
    "            file_path = os.path.join(sev_path, file)\n",
    "            gfcc = extract_gfcc(file_path)\n",
    "\n",
    "            save_file = os.path.join(save_sev_path, file.replace(\".wav\", \".npy\"))\n",
    "            np.save(save_file, gfcc)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "28451fba",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:32:55.054596Z",
     "iopub.status.busy": "2026-05-25T07:32:55.054256Z",
     "iopub.status.idle": "2026-05-25T07:55:39.048041Z",
     "shell.execute_reply": "2026-05-25T07:55:39.047245Z"
    },
    "papermill": {
     "duration": 1364.000529,
     "end_time": "2026-05-25T07:55:39.049978+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:32:55.049449+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train-N: 100%|██████████| 7273/7273 [12:32<00:00,  9.66it/s]\n",
      "train-H: 100%|██████████| 99/99 [00:13<00:00,  7.56it/s]\n",
      "train-M: 100%|██████████| 2062/2062 [03:42<00:00,  9.27it/s]\n",
      "train-L: 100%|██████████| 817/817 [01:32<00:00,  8.84it/s]\n",
      "test-N: 100%|██████████| 2037/2037 [03:33<00:00,  9.55it/s]\n",
      "test-H: 100%|██████████| 59/59 [00:06<00:00,  9.06it/s]\n",
      "test-M: 100%|██████████| 506/506 [00:54<00:00,  9.36it/s]\n",
      "test-L: 100%|██████████| 79/79 [00:09<00:00,  8.67it/s]\n"
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
   "id": "0098cbc3",
   "metadata": {
    "papermill": {
     "duration": 0.314595,
     "end_time": "2026-05-25T07:55:39.679091+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:39.364496+00:00",
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
   "id": "c6e564fa",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:40.296926Z",
     "iopub.status.busy": "2026-05-25T07:55:40.296116Z",
     "iopub.status.idle": "2026-05-25T07:55:40.302856Z",
     "shell.execute_reply": "2026-05-25T07:55:40.302245Z"
    },
    "papermill": {
     "duration": 0.317049,
     "end_time": "2026-05-25T07:55:40.304303+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:39.987254+00:00",
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
   "id": "258809ec",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:41.055369Z",
     "iopub.status.busy": "2026-05-25T07:55:41.054832Z",
     "iopub.status.idle": "2026-05-25T07:55:41.079981Z",
     "shell.execute_reply": "2026-05-25T07:55:41.079216Z"
    },
    "papermill": {
     "duration": 0.463839,
     "end_time": "2026-05-25T07:55:41.081816+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:40.617977+00:00",
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
   "id": "923e836e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:41.814688Z",
     "iopub.status.busy": "2026-05-25T07:55:41.814381Z",
     "iopub.status.idle": "2026-05-25T07:55:41.879197Z",
     "shell.execute_reply": "2026-05-25T07:55:41.878466Z"
    },
    "papermill": {
     "duration": 0.444126,
     "end_time": "2026-05-25T07:55:41.880975+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:41.436849+00:00",
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
   "id": "45a1297b",
   "metadata": {
    "papermill": {
     "duration": 0.433115,
     "end_time": "2026-05-25T07:55:42.644699+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:42.211584+00:00",
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
   "id": "44afd0c0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:43.308439Z",
     "iopub.status.busy": "2026-05-25T07:55:43.307741Z",
     "iopub.status.idle": "2026-05-25T07:55:43.314721Z",
     "shell.execute_reply": "2026-05-25T07:55:43.314000Z"
    },
    "papermill": {
     "duration": 0.33842,
     "end_time": "2026-05-25T07:55:43.316342+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:42.977922+00:00",
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
   "id": "53940d73",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:43.966599Z",
     "iopub.status.busy": "2026-05-25T07:55:43.965790Z",
     "iopub.status.idle": "2026-05-25T07:55:51.970061Z",
     "shell.execute_reply": "2026-05-25T07:55:51.969012Z"
    },
    "papermill": {
     "duration": 8.327461,
     "end_time": "2026-05-25T07:55:51.971994+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:43.644533+00:00",
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
   "id": "a0ef2614",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:52.734663Z",
     "iopub.status.busy": "2026-05-25T07:55:52.733920Z",
     "iopub.status.idle": "2026-05-25T07:55:52.738250Z",
     "shell.execute_reply": "2026-05-25T07:55:52.737520Z"
    },
    "papermill": {
     "duration": 0.446391,
     "end_time": "2026-05-25T07:55:52.739741+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:52.293350+00:00",
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
   "id": "8ebabce5",
   "metadata": {
    "papermill": {
     "duration": 0.327981,
     "end_time": "2026-05-25T07:55:53.395992+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:53.068011+00:00",
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
   "id": "826f200a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:54.040991Z",
     "iopub.status.busy": "2026-05-25T07:55:54.040223Z",
     "iopub.status.idle": "2026-05-25T07:55:54.043877Z",
     "shell.execute_reply": "2026-05-25T07:55:54.043180Z"
    },
    "papermill": {
     "duration": 0.326173,
     "end_time": "2026-05-25T07:55:54.045379+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:53.719206+00:00",
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
   "id": "1862b11b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:54.793314Z",
     "iopub.status.busy": "2026-05-25T07:55:54.792485Z",
     "iopub.status.idle": "2026-05-25T07:55:54.797490Z",
     "shell.execute_reply": "2026-05-25T07:55:54.796555Z"
    },
    "papermill": {
     "duration": 0.332128,
     "end_time": "2026-05-25T07:55:54.798988+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:54.466860+00:00",
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
   "id": "7ce75ba3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:55:55.456713Z",
     "iopub.status.busy": "2026-05-25T07:55:55.455799Z",
     "iopub.status.idle": "2026-05-25T07:58:46.288720Z",
     "shell.execute_reply": "2026-05-25T07:58:46.287783Z"
    },
    "papermill": {
     "duration": 171.163493,
     "end_time": "2026-05-25T07:58:46.290536+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:55:55.127043+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:10<00:00, 28.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20: Loss=156.3918, Val Acc=0.9034\n",
      "✅ Best model updated! Val Acc: 0.9034\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.80it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/20: Loss=66.8947, Val Acc=0.9346\n",
      "✅ Best model updated! Val Acc: 0.9346\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.31it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/20: Loss=45.3463, Val Acc=0.9541\n",
      "✅ Best model updated! Val Acc: 0.9541\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.34it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/20: Loss=37.8660, Val Acc=0.9580\n",
      "✅ Best model updated! Val Acc: 0.9580\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.30it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/20: Loss=30.9994, Val Acc=0.9444\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 37.13it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/20: Loss=22.5110, Val Acc=0.9746\n",
      "✅ Best model updated! Val Acc: 0.9746\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.96it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/20: Loss=16.4915, Val Acc=0.9727\n"
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
      "Epoch 8/20: Loss=12.2567, Val Acc=0.9785\n",
      "✅ Best model updated! Val Acc: 0.9785\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.53it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/20: Loss=10.5860, Val Acc=0.9659\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:07<00:00, 36.30it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/20: Loss=7.5363, Val Acc=0.9346\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.98it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/20: Loss=6.0792, Val Acc=0.9727\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.91it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/20: Loss=8.3872, Val Acc=0.9698\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.39it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/20: Loss=10.1655, Val Acc=0.9668\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.22it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/20: Loss=4.9126, Val Acc=0.9688\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 34.95it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/20: Loss=6.4137, Val Acc=0.9717\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 34.92it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/20: Loss=5.0072, Val Acc=0.9756\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 34.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/20: Loss=2.5313, Val Acc=0.9766\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 34.56it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 18/20: Loss=0.6154, Val Acc=0.9834\n",
      "✅ Best model updated! Val Acc: 0.9834\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 34.89it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 19/20: Loss=0.0714, Val Acc=0.9746\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 289/289 [00:08<00:00, 35.46it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 20/20: Loss=0.0754, Val Acc=0.9795\n",
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
   "id": "c96f5f40",
   "metadata": {
    "papermill": {
     "duration": 0.373903,
     "end_time": "2026-05-25T07:58:47.136736+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:46.762833+00:00",
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
   "id": "6e99eb18",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:47.884735Z",
     "iopub.status.busy": "2026-05-25T07:58:47.884428Z",
     "iopub.status.idle": "2026-05-25T07:58:47.889919Z",
     "shell.execute_reply": "2026-05-25T07:58:47.889246Z"
    },
    "papermill": {
     "duration": 0.379859,
     "end_time": "2026-05-25T07:58:47.891533+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:47.511674+00:00",
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
   "id": "1843ab50",
   "metadata": {
    "papermill": {
     "duration": 0.370628,
     "end_time": "2026-05-25T07:58:48.727399+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:48.356771+00:00",
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
   "id": "511b2dca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:49.482647Z",
     "iopub.status.busy": "2026-05-25T07:58:49.481823Z",
     "iopub.status.idle": "2026-05-25T07:58:49.486121Z",
     "shell.execute_reply": "2026-05-25T07:58:49.485414Z"
    },
    "papermill": {
     "duration": 0.383045,
     "end_time": "2026-05-25T07:58:49.487631+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:49.104586+00:00",
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
   "id": "471efde5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:50.334514Z",
     "iopub.status.busy": "2026-05-25T07:58:50.333780Z",
     "iopub.status.idle": "2026-05-25T07:58:50.365523Z",
     "shell.execute_reply": "2026-05-25T07:58:50.364634Z"
    },
    "papermill": {
     "duration": 0.417985,
     "end_time": "2026-05-25T07:58:50.367066+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:49.949081+00:00",
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
   "id": "ed7b6ac3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:51.124330Z",
     "iopub.status.busy": "2026-05-25T07:58:51.123455Z",
     "iopub.status.idle": "2026-05-25T07:58:52.337512Z",
     "shell.execute_reply": "2026-05-25T07:58:52.336491Z"
    },
    "papermill": {
     "duration": 1.595599,
     "end_time": "2026-05-25T07:58:52.339027+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:50.743428+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 84/84 [00:01<00:00, 69.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Test Accuracy: 0.8952\n"
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
   "id": "b67cb278",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:53.180676Z",
     "iopub.status.busy": "2026-05-25T07:58:53.180238Z",
     "iopub.status.idle": "2026-05-25T07:58:55.389560Z",
     "shell.execute_reply": "2026-05-25T07:58:55.388671Z"
    },
    "papermill": {
     "duration": 2.588478,
     "end_time": "2026-05-25T07:58:55.392096+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:52.803618+00:00",
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
      "           L       0.17      0.39      0.24        79\n",
      "           M       0.78      0.66      0.72       506\n",
      "           H       0.11      0.02      0.03        59\n",
      "\n",
      "    accuracy                           0.90      2681\n",
      "   macro avg       0.51      0.52      0.49      2681\n",
      "weighted avg       0.90      0.90      0.90      2681\n",
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
   "id": "accd87ba",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:56.173880Z",
     "iopub.status.busy": "2026-05-25T07:58:56.173381Z",
     "iopub.status.idle": "2026-05-25T07:58:58.804261Z",
     "shell.execute_reply": "2026-05-25T07:58:58.803111Z"
    },
    "papermill": {
     "duration": 3.02792,
     "end_time": "2026-05-25T07:58:58.806268+00:00",
     "exception": false,
     "start_time": "2026-05-25T07:58:55.778348+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 84/84 [00:01<00:00, 69.86it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Saved results.csv and results.xlsx\n",
      "                   filename    score_N    score_L   score_M    score_H  \\\n",
      "0  CM27_Gujarati_N_0011.npy  24.052917  -1.461073 -6.784083 -42.326771   \n",
      "1   CM35_English_N_0022.npy  32.946205 -12.823582 -7.841996 -52.483746   \n",
      "2   CF04_English_N_0095.npy  41.390793 -31.679148 -7.341860 -56.807354   \n",
      "3   CM35_English_N_0124.npy  25.186026 -11.488960 -7.775810 -36.218849   \n",
      "4     CF09_Hindi_N_0169.npy  28.586273 -17.345211 -4.129201 -41.612835   \n",
      "\n",
      "   prob_N        prob_L        prob_M        prob_H predict class actual class  \n",
      "0     1.0  8.306436e-12  4.051915e-14  1.484805e-29             N            N  \n",
      "1     1.0  1.325666e-20  1.931566e-18  7.911256e-38             N            N  \n",
      "2     1.0  1.845563e-32  6.849813e-22  2.256091e-43             N            N  \n",
      "3     1.0  1.181014e-16  4.840116e-15  2.148824e-27             N            N  \n",
      "4     1.0  1.127742e-20  6.192279e-15  3.257715e-31             N            N  \n"
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
   "execution_count": 22,
   "id": "091bf294",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-05-25T07:58:59.675770Z",
     "iopub.status.busy": "2026-05-25T07:58:59.674605Z",
     "iopub.status.idle": "2026-05-25T07:58:59.684528Z",
     "shell.execute_reply": "2026-05-25T07:58:59.683523Z"
    },
    "papermill": {
     "duration": 0.391064,
     "end_time": "2026-05-25T07:58:59.685757+00:00",
     "exception": true,
     "start_time": "2026-05-25T07:58:59.294693+00:00",
     "status": "failed"
    },
    "tags": []
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "unterminated string literal (detected at line 58) (2114917303.py, line 58)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"/tmp/ipykernel_23/2114917303.py\"\u001b[0;36m, line \u001b[0;32m58\u001b[0m\n\u001b[0;31m    print(\"✅ Saved distribution_plot.png\u001b[0m\n\u001b[0m          ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m unterminated string literal (detected at line 58)\n"
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
    "print(\"✅ Saved distribution_plot.png\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa297609",
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
    "## Single File Prediction (Optional)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d62c991",
   "metadata": {
    "execution": {
     "iopub.status.busy": "2026-05-25T07:30:13.817141Z",
     "iopub.status.idle": "2026-05-25T07:30:13.817483Z",
     "shell.execute_reply": "2026-05-25T07:30:13.817367Z",
     "shell.execute_reply.started": "2026-05-25T07:30:13.817340Z"
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
   "duration": 1581.934735,
   "end_time": "2026-05-25T07:59:02.887456+00:00",
   "environment_variables": {},
   "exception": true,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-05-25T07:32:40.952721+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
