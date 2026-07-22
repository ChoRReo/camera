# 偏振相机数据处理脚本

这个目录包含三个脚本，用于处理偏振相机采集的数据：

1. 对每个偏振方向的多张图像取平均，降低随机噪声。
2. 根据六个偏振方向的平均图计算 DoLP、DoCP、AoP。
3. 比较不同采集张数下的 DoLP、DoCP、AoP 稳定性，用来判断是否需要采集 1000 张。

## 数据结构

原始数据默认结构如下：

```text
/Volumes/xuyifan_u/cam_data/
  00001/
    0/
    45/
    90/
    135/
    L/
    R/
  00002/
    0/
    45/
    90/
    135/
    L/
    R/
```

其中 `00001`、`00002` 表示不同场景；每个场景下有六个偏振方向，每个方向内包含多张 PNG 图像。

## 1. 计算六个方向的平均图

只处理一个场景时：

```bash
python3 process_cam_data.py /Volumes/xuyifan_u/cam_data -o avg_00001 --scenes 00001 --overwrite
```

输出：

```text
avg_00001/0.png
avg_00001/45.png
avg_00001/90.png
avg_00001/135.png
avg_00001/L.png
avg_00001/R.png
avg_00001/cam_data_average_summary.json
```

同时处理多个场景时：

```bash
python3 process_cam_data.py /Volumes/xuyifan_u/cam_data -o avg_all --overwrite
```

输出会按场景分文件夹，避免文件名覆盖：

```text
avg_all/00001/0.png
avg_all/00001/45.png
...
avg_all/00002/0.png
avg_all/00002/45.png
...
```

`cam_data_average_summary.json` 中包含随机噪声估计和推荐采集张数。

## 2. 计算 DoLP、DoCP、AoP

对单个场景的平均图计算：

```bash
python3 compute_polarization_maps.py avg_00001 -o pol_00001 --overwrite
```

输出：

```text
pol_00001/dolp.npy
pol_00001/dolp.png
pol_00001/docp.npy
pol_00001/docp.png
pol_00001/aop_rad.npy
pol_00001/aop_rad.png
pol_00001/polarization_maps_summary.json
```

DoLP、DoCP、AoP 默认按 RGB 三通道分别计算。`.npy` 保存 float32 原始数值，形状通常为 `H x W x 3`，适合后续计算；`.png` 是 RGB 可视化结果。

默认计算公式：

```text
S0 = 0.5 * ((I0 + I90) + (I45 + I135))
S1 = I0 - I90
S2 = I45 - I135
S3 = IR - IL

DoLP = sqrt(S1^2 + S2^2) / S0
DoCP = S3 / S0
AoP  = 0.5 * atan2(S2, S1), range [0, pi)
```

如果圆偏振正负方向定义相反，使用：

```bash
python3 compute_polarization_maps.py avg_00001 -o pol_00001 --circular-sign L-minus-R --overwrite
```

## 3. 比较不同采集张数

这个脚本直接读取原始数据，不需要先生成平均图。它会分别用不同张数计算平均图、DoLP、DoCP、AoP，然后和全量参考结果比较。

比较过程同样默认按 RGB 三通道分别计算偏振参数。

```bash
python3 compare_frame_counts.py /Volumes/xuyifan_u/cam_data -o compare_result --scenes 00001 --overwrite
```

指定要比较的张数：

```bash
python3 compare_frame_counts.py /Volumes/xuyifan_u/cam_data \
  -o compare_result \
  --scenes 00001 \
  --counts 16 32 64 100 150 200 300 \
  --overwrite
```

输出：

```text
compare_result/frame_count_comparison.csv
compare_result/frame_count_comparison_summary.json
```

CSV 中重点查看：

```text
dolp_rms
docp_rms
aop_rms_deg
passes_thresholds
```

默认认为满足以下条件时采集张数已经够用：

```text
DoLP RMS error <= 0.005
DoCP RMS error <= 0.005
AoP RMS error <= 1 degree
```

## 依赖

需要 Python 3，以及：

```bash
pip install numpy pillow
```

## 建议流程

先用 `compare_frame_counts.py` 判断 32、64、100、200 张等采集数量是否已经足够稳定。确定合适张数后，再用 `process_cam_data.py` 生成六个方向的平均图，最后用 `compute_polarization_maps.py` 计算偏振参数。
