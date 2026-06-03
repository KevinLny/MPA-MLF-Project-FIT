# 60 GHz Radar Signal Classification via Deep Learning

## Project Overview
This repository contains the final project for the **MPA-MLF (Machine Learning)** course completed during an Erasmus+ semester at the **Brno University of Technology**. 

The goal of this project is to accurately classify room occupancy (from 0 to 3 people) utilizing **60 GHz radar signal transmissions**. By processing multi-dimensional reflections, the system effectively monitors environments without compromising privacy, making it a robust alternative to traditional camera-based tracking.

---

## 🛰️ Problem Statement & Dataset
The system analyzes snapshots of radar signals in the **delay-Doppler domain**, which represent reflections from moving targets at specific distances and velocities:
* **Doppler frequency shifts** correlate with the speed and direction of target motion.
* **Signal delays** correlate with the distance of the targets from the receiver.

### Target Classes:
1. **Machine only** (0 persons in the room)
2. **One person** present
3. **Two persons** present
4. **Three persons** present

The challenge involves handling real-world signal distortions, environmental noise, reflections, and missed targets embedded within 2D `.png` domain snapshots.
