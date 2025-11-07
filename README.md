# RTPs-Distribution-Finder

Python package to **detect and plot Rapid Transition Events (RTPs)** in continuous signals.

📄 **Reference:** [ArXiv:2506.06168](https://arxiv.org/abs/2506.06168)

---

## 🔍 Overview

`RTPs-Distribution-Finder` provides simple tools to:
- Detect rapid transitions in continuous or noisy signals  
- Visualize RTPs as a simple raster plot 

---

## ⚙️ Installation

git clone https://github.com/yourusername/RTPs-Distribution-Finder.git
cd RTPs-Distribution-Finder
pip install -r requirements.txt

from rtps_distribution_finder import RTPFinder, plot_rtp_distribution

## 🧠 Quick Example

signal = your_signal_array
rtp = RTPFinder(signal, sampling_rate=1000)

events = rtp.detect(threshold=0.05)
plot_rtp_distribution(events)




