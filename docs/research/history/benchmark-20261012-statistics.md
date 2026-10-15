> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

# Benchmark 统计报告

正式统计范围为 114 个锁定分类任务；15 个开发任务仅用于确定固定模型，不进入正式效应统计。
正式 suite：biology5, raman34, tabmini25, ucr85。

R 每轮从当轮合法模型均匀抽取；balanced accuracy 与 accuracy 报告解析期望，多轮 Macro-F1 按固定 seed 模拟。P 仅依据 validation_score 选模型，Test 只用于计算最终分数。

## 方法结果

| 方法 | 预算 | 完整任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| R0 | 1 | 114/114 | 0.6692 | 0.6561 | 0.7145 |
| R1 | 3 | 114/114 | 0.6882 | 0.6775 | 0.7300 |
| A0 | 1 | 114/114 | 0.6802 | 0.6637 | 0.7224 |
| A1 | 3 | 114/114 | 0.6861 | 0.6700 | 0.7282 |
| F0 | 1 | 114/114 | 0.6718 | 0.6585 | 0.7200 |
| F1 | 3 | 114/114 | 0.6757 | 0.6623 | 0.7229 |
| P0 | 1 | 114/114 | 0.7543 | 0.7468 | 0.7886 |
| P1 | 3 | 114/114 | 0.7577 | 0.7506 | 0.7879 |

## 按 suite 汇总

| 方法 | 预算 | suite | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| R0 | 1 | biology5 | 5/5 | 0.7625 | 0.7749 | 0.7900 |
| R0 | 1 | raman34 | 17/17 | 0.6926 | 0.6857 | 0.7090 |
| R0 | 1 | tabmini25 | 20/20 | 0.7110 | 0.7085 | 0.7804 |
| R0 | 1 | ucr85 | 72/72 | 0.6455 | 0.6263 | 0.6922 |
| R1 | 3 | biology5 | 5/5 | 0.8099 | 0.8186 | 0.8280 |
| R1 | 3 | raman34 | 17/17 | 0.7185 | 0.7111 | 0.7323 |
| R1 | 3 | tabmini25 | 20/20 | 0.7140 | 0.7116 | 0.7825 |
| R1 | 3 | ucr85 | 72/72 | 0.6655 | 0.6504 | 0.7082 |
| A0 | 1 | biology5 | 5/5 | 0.8151 | 0.8221 | 0.8375 |
| A0 | 1 | raman34 | 17/17 | 0.7641 | 0.7315 | 0.7867 |
| A0 | 1 | tabmini25 | 20/20 | 0.7235 | 0.7194 | 0.7958 |
| A0 | 1 | ucr85 | 72/72 | 0.6389 | 0.6212 | 0.6788 |
| A1 | 3 | biology5 | 5/5 | 0.8048 | 0.8144 | 0.8348 |
| A1 | 3 | raman34 | 17/17 | 0.7695 | 0.7359 | 0.7900 |
| A1 | 3 | tabmini25 | 20/20 | 0.7084 | 0.7024 | 0.7853 |
| A1 | 3 | ucr85 | 72/72 | 0.6519 | 0.6354 | 0.6903 |
| F0 | 1 | biology5 | 5/5 | 0.8464 | 0.8509 | 0.8553 |
| F0 | 1 | raman34 | 17/17 | 0.7844 | 0.7726 | 0.8013 |
| F0 | 1 | tabmini25 | 20/20 | 0.6977 | 0.6903 | 0.7740 |
| F0 | 1 | ucr85 | 72/72 | 0.6259 | 0.6094 | 0.6764 |
| F1 | 3 | biology5 | 5/5 | 0.8464 | 0.8509 | 0.8553 |
| F1 | 3 | raman34 | 17/17 | 0.7854 | 0.7739 | 0.8021 |
| F1 | 3 | tabmini25 | 20/20 | 0.7036 | 0.6958 | 0.7777 |
| F1 | 3 | ucr85 | 72/72 | 0.6303 | 0.6136 | 0.6798 |
| P0 | 1 | biology5 | 5/5 | 0.8317 | 0.8381 | 0.8385 |
| P0 | 1 | raman34 | 17/17 | 0.7724 | 0.7612 | 0.7829 |
| P0 | 1 | tabmini25 | 20/20 | 0.7493 | 0.7476 | 0.8030 |
| P0 | 1 | ucr85 | 72/72 | 0.7460 | 0.7368 | 0.7825 |
| P1 | 3 | biology5 | 5/5 | 0.8386 | 0.8408 | 0.8415 |
| P1 | 3 | raman34 | 17/17 | 0.7533 | 0.7473 | 0.7606 |
| P1 | 3 | tabmini25 | 20/20 | 0.7543 | 0.7530 | 0.8061 |
| P1 | 3 | ucr85 | 72/72 | 0.7541 | 0.7445 | 0.7856 |

## 按 family 汇总

| 方法 | 预算 | family_id | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| R0 | 1 | biology5:ALLAML | 1/1 | 0.9063 | 0.9158 | 0.9266 |
| R0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.7648 | 0.7712 | 0.7104 |
| R0 | 1 | biology5:Carcinom | 1/1 | 0.6597 | 0.6873 | 0.7537 |
| R0 | 1 | biology5:GLIOMA | 1/1 | 0.7218 | 0.7314 | 0.7600 |
| R0 | 1 | biology5:colon | 1/1 | 0.7599 | 0.7687 | 0.7995 |
| R0 | 1 | raman34:cancer_cell_media | 3/3 | 0.7150 | 0.7176 | 0.7199 |
| R0 | 1 | raman34:chlorinated_samples | 1/1 | 0.8635 | 0.8704 | 0.8910 |
| R0 | 1 | raman34:comfile_stroke | 1/1 | 0.7962 | 0.7944 | 0.7962 |
| R0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 0.8974 | 0.8940 | 0.8974 |
| R0 | 1 | raman34:diabetes_skin | 4/4 | 0.4870 | 0.4588 | 0.5077 |
| R0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.8642 | 0.8881 | 0.9046 |
| R0 | 1 | raman34:head_neck_cancer | 1/1 | 0.3952 | 0.3715 | 0.4664 |
| R0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.8346 | 0.8343 | 0.8346 |
| R0 | 1 | raman34:marine_pathogens | 2/2 | 0.9495 | 0.9510 | 0.9522 |
| R0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.8668 | 0.8833 | 0.8684 |
| R0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.2637 | 0.2312 | 0.2986 |
| R0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.8049 | 0.8027 | 0.8038 |
| R0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8457 | 0.8454 | 0.8457 |
| R0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.7392 | 0.7601 | 0.8286 |
| R0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.5673 | 0.5557 | 0.6871 |
| R0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8077 | 0.8083 | 0.8104 |
| R0 | 1 | tabmini25:backache | 1/1 | 0.5176 | 0.5065 | 0.8460 |
| R0 | 1 | tabmini25:biomed | 1/1 | 0.8343 | 0.8473 | 0.8660 |
| R0 | 1 | tabmini25:breast_cancer | 1/1 | 0.5859 | 0.5866 | 0.6988 |
| R0 | 1 | tabmini25:bupa | 1/1 | 0.6137 | 0.6115 | 0.6124 |
| R0 | 1 | tabmini25:corral | 1/1 | 0.8958 | 0.8977 | 0.8991 |
| R0 | 1 | tabmini25:haberman | 1/1 | 0.5358 | 0.5091 | 0.7381 |
| R0 | 1 | tabmini25:hepatitis | 1/1 | 0.6507 | 0.6710 | 0.8230 |
| R0 | 1 | tabmini25:irish | 1/1 | 0.8475 | 0.8488 | 0.8509 |
| R0 | 1 | tabmini25:labor | 1/1 | 0.8157 | 0.8201 | 0.8396 |
| R0 | 1 | tabmini25:lupus | 1/1 | 0.7449 | 0.7496 | 0.7652 |
| R0 | 1 | tabmini25:molecular_biology_promoters | 1/1 | 0.8019 | 0.8016 | 0.8019 |
| R0 | 1 | tabmini25:mux6 | 1/1 | 0.6161 | 0.6154 | 0.6161 |
| R0 | 1 | tabmini25:postoperative_patient_data | 1/1 | 0.4970 | 0.4277 | 0.7175 |
| R0 | 1 | tabmini25:prnn_synth | 1/1 | 0.8491 | 0.8491 | 0.8491 |
| R0 | 1 | tabmini25:saheart | 1/1 | 0.6495 | 0.6554 | 0.7084 |
| R0 | 1 | ucr85:Adiac | 1/1 | 0.6105 | 0.5697 | 0.5951 |
| R0 | 1 | ucr85:ArrowHead | 1/1 | 0.6467 | 0.6362 | 0.6404 |
| R0 | 1 | ucr85:CBF | 1/1 | 0.8230 | 0.8176 | 0.8224 |
| R0 | 1 | ucr85:Car | 1/1 | 0.6021 | 0.5857 | 0.6038 |
| R0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.6362 | 0.6262 | 0.7116 |
| R0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.5592 | 0.5515 | 0.5581 |
| R0 | 1 | ucr85:Computers | 1/1 | 0.6283 | 0.6258 | 0.6283 |
| R0 | 1 | ucr85:Cricket | 3/3 | 0.5221 | 0.4998 | 0.5179 |
| R0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5744 | 0.5568 | 0.7919 |
| R0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7252 | 0.7274 | 0.7503 |
| R0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4393 | 0.4055 | 0.7552 |
| R0 | 1 | ucr85:ECG200 | 1/1 | 0.8062 | 0.8104 | 0.8285 |
| R0 | 1 | ucr85:ECG5000 | 1/1 | 0.4283 | 0.4280 | 0.9163 |
| R0 | 1 | ucr85:Earthquakes | 1/1 | 0.5124 | 0.4775 | 0.7860 |
| R0 | 1 | ucr85:ElectricDevices | 1/1 | 0.5001 | 0.4838 | 0.6118 |
| R0 | 1 | ucr85:FISH | 1/1 | 0.7250 | 0.7043 | 0.7231 |
| R0 | 1 | ucr85:FaceAll | 1/1 | 0.7541 | 0.6878 | 0.7067 |
| R0 | 1 | ucr85:FaceFour | 1/1 | 0.6302 | 0.5604 | 0.6128 |
| R0 | 1 | ucr85:FacesUCR | 1/1 | 0.5939 | 0.5902 | 0.6609 |
| R0 | 1 | ucr85:FordA | 1/1 | 0.7297 | 0.7087 | 0.7321 |
| R0 | 1 | ucr85:FordB | 1/1 | 0.7044 | 0.7031 | 0.7041 |
| R0 | 1 | ucr85:Gun_Point | 1/1 | 0.8619 | 0.8615 | 0.8621 |
| R0 | 1 | ucr85:Ham | 1/1 | 0.6382 | 0.6218 | 0.6418 |
| R0 | 1 | ucr85:HandOutlines | 1/1 | 0.8119 | 0.8175 | 0.8345 |
| R0 | 1 | ucr85:Haptics | 1/1 | 0.3852 | 0.3488 | 0.3906 |
| R0 | 1 | ucr85:Herring | 1/1 | 0.5326 | 0.4548 | 0.5781 |
| R0 | 1 | ucr85:InlineSkate | 1/1 | 0.3082 | 0.2899 | 0.3092 |
| R0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.5378 | 0.5234 | 0.5378 |
| R0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9438 | 0.9438 | 0.9438 |
| R0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.6205 | 0.6192 | 0.6205 |
| R0 | 1 | ucr85:Lighting2 | 1/1 | 0.6919 | 0.6873 | 0.7049 |
| R0 | 1 | ucr85:Lighting7 | 1/1 | 0.5374 | 0.4753 | 0.5648 |
| R0 | 1 | ucr85:MALLAT | 1/1 | 0.7380 | 0.6960 | 0.7361 |
| R0 | 1 | ucr85:MedicalImages | 1/1 | 0.3680 | 0.3623 | 0.6041 |
| R0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.5745 | 0.5165 | 0.6706 |
| R0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5811 | 0.5303 | 0.5312 |
| R0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3644 | 0.3235 | 0.6154 |
| R0 | 1 | ucr85:MoteStrain | 1/1 | 0.7340 | 0.6973 | 0.7407 |
| R0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.7818 | 0.7656 | 0.7813 |
| R0 | 1 | ucr85:OSULeaf | 1/1 | 0.6382 | 0.6195 | 0.6491 |
| R0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.6928 | 0.6767 | 0.7331 |
| R0 | 1 | ucr85:Plane | 1/1 | 0.9465 | 0.9386 | 0.9436 |
| R0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7705 | 0.7437 | 0.8338 |
| R0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7888 | 0.8034 | 0.8467 |
| R0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.4473 | 0.4219 | 0.7898 |
| R0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.4576 | 0.4509 | 0.4576 |
| R0 | 1 | ucr85:ScreenType | 1/1 | 0.4624 | 0.4582 | 0.4624 |
| R0 | 1 | ucr85:ShapeletSim | 1/1 | 0.6295 | 0.6054 | 0.6295 |
| R0 | 1 | ucr85:ShapesAll | 1/1 | 0.6476 | 0.6185 | 0.6476 |
| R0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.6098 | 0.5988 | 0.6098 |
| R0 | 1 | ucr85:StarLightCurves | 1/1 | 0.8826 | 0.8793 | 0.9293 |
| R0 | 1 | ucr85:Strawberry | 1/1 | 0.9114 | 0.9060 | 0.9209 |
| R0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.8179 | 0.8036 | 0.8130 |
| R0 | 1 | ucr85:Symbols | 1/1 | 0.7336 | 0.7146 | 0.7342 |
| R0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.7320 | 0.7298 | 0.7301 |
| R0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.7040 | 0.6455 | 0.7053 |
| R0 | 1 | ucr85:Trace | 1/1 | 0.8767 | 0.8442 | 0.8638 |
| R0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.7364 | 0.7190 | 0.7363 |
| R0 | 1 | ucr85:Two_Patterns | 1/1 | 0.8134 | 0.8086 | 0.8136 |
| R0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.1974 | 0.1685 | 0.4236 |
| R0 | 1 | ucr85:Worms | 1/1 | 0.3358 | 0.3105 | 0.4450 |
| R0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5892 | 0.5740 | 0.6294 |
| R0 | 1 | ucr85:synthetic_control | 1/1 | 0.9126 | 0.9072 | 0.9126 |
| R0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.7035 | 0.6842 | 0.7028 |
| R0 | 1 | ucr85:wafer | 1/1 | 0.6441 | 0.6581 | 0.9211 |
| R0 | 1 | ucr85:yoga | 1/1 | 0.6969 | 0.6953 | 0.7033 |
| R1 | 3 | biology5:ALLAML | 1/1 | 0.9332 | 0.9346 | 0.9425 |
| R1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.7939 | 0.7988 | 0.7362 |
| R1 | 3 | biology5:Carcinom | 1/1 | 0.7853 | 0.8057 | 0.8571 |
| R1 | 3 | biology5:GLIOMA | 1/1 | 0.7367 | 0.7468 | 0.7771 |
| R1 | 3 | biology5:colon | 1/1 | 0.8003 | 0.8069 | 0.8272 |
| R1 | 3 | raman34:cancer_cell_media | 3/3 | 0.7799 | 0.7820 | 0.7838 |
| R1 | 3 | raman34:chlorinated_samples | 1/1 | 0.8991 | 0.9037 | 0.9157 |
| R1 | 3 | raman34:comfile_stroke | 1/1 | 0.7923 | 0.7920 | 0.7923 |
| R1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 0.9026 | 0.8991 | 0.9026 |
| R1 | 3 | raman34:diabetes_skin | 4/4 | 0.4833 | 0.4569 | 0.5038 |
| R1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9457 | 0.9563 | 0.9612 |
| R1 | 3 | raman34:head_neck_cancer | 1/1 | 0.4284 | 0.4136 | 0.5017 |
| R1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.8681 | 0.8679 | 0.8681 |
| R1 | 3 | raman34:marine_pathogens | 2/2 | 0.9693 | 0.9694 | 0.9700 |
| R1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9190 | 0.9300 | 0.9196 |
| R1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2476 | 0.2138 | 0.2805 |
| R1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8083 | 0.8059 | 0.8072 |
| R1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8400 | 0.8401 | 0.8400 |
| R1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.7654 | 0.7805 | 0.8400 |
| R1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5623 | 0.5526 | 0.6803 |
| R1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8020 | 0.8017 | 0.8049 |
| R1 | 3 | tabmini25:backache | 1/1 | 0.5098 | 0.4989 | 0.8325 |
| R1 | 3 | tabmini25:biomed | 1/1 | 0.8348 | 0.8482 | 0.8667 |
| R1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5925 | 0.5937 | 0.7238 |
| R1 | 3 | tabmini25:bupa | 1/1 | 0.6125 | 0.6102 | 0.6112 |
| R1 | 3 | tabmini25:corral | 1/1 | 0.9037 | 0.9050 | 0.9062 |
| R1 | 3 | tabmini25:haberman | 1/1 | 0.5351 | 0.5136 | 0.7297 |
| R1 | 3 | tabmini25:hepatitis | 1/1 | 0.6607 | 0.6806 | 0.8258 |
| R1 | 3 | tabmini25:irish | 1/1 | 0.8500 | 0.8505 | 0.8534 |
| R1 | 3 | tabmini25:labor | 1/1 | 0.8400 | 0.8473 | 0.8647 |
| R1 | 3 | tabmini25:lupus | 1/1 | 0.7326 | 0.7369 | 0.7537 |
| R1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.8046 | 0.8051 | 0.8046 |
| R1 | 3 | tabmini25:mux6 | 1/1 | 0.6339 | 0.6328 | 0.6339 |
| R1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.4970 | 0.4279 | 0.7175 |
| R1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8486 | 0.8486 | 0.8486 |
| R1 | 3 | tabmini25:saheart | 1/1 | 0.6463 | 0.6519 | 0.7044 |
| R1 | 3 | ucr85:Adiac | 1/1 | 0.6696 | 0.6321 | 0.6577 |
| R1 | 3 | ucr85:ArrowHead | 1/1 | 0.6425 | 0.6330 | 0.6369 |
| R1 | 3 | ucr85:CBF | 1/1 | 0.8382 | 0.8364 | 0.8378 |
| R1 | 3 | ucr85:Car | 1/1 | 0.6220 | 0.6058 | 0.6231 |
| R1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.6786 | 0.6833 | 0.7455 |
| R1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.5654 | 0.5579 | 0.5644 |
| R1 | 3 | ucr85:Computers | 1/1 | 0.6295 | 0.6288 | 0.6295 |
| R1 | 3 | ucr85:Cricket | 3/3 | 0.5474 | 0.5355 | 0.5443 |
| R1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5912 | 0.5770 | 0.7921 |
| R1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7356 | 0.7355 | 0.7535 |
| R1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4528 | 0.4261 | 0.7606 |
| R1 | 3 | ucr85:ECG200 | 1/1 | 0.7877 | 0.7912 | 0.8108 |
| R1 | 3 | ucr85:ECG5000 | 1/1 | 0.4540 | 0.4529 | 0.9261 |
| R1 | 3 | ucr85:Earthquakes | 1/1 | 0.5558 | 0.5140 | 0.7391 |
| R1 | 3 | ucr85:ElectricDevices | 1/1 | 0.5107 | 0.5033 | 0.6180 |
| R1 | 3 | ucr85:FISH | 1/1 | 0.7631 | 0.7533 | 0.7618 |
| R1 | 3 | ucr85:FaceAll | 1/1 | 0.8018 | 0.7397 | 0.7520 |
| R1 | 3 | ucr85:FaceFour | 1/1 | 0.6330 | 0.5616 | 0.6145 |
| R1 | 3 | ucr85:FacesUCR | 1/1 | 0.6552 | 0.6563 | 0.7209 |
| R1 | 3 | ucr85:FordA | 1/1 | 0.7471 | 0.7346 | 0.7486 |
| R1 | 3 | ucr85:FordB | 1/1 | 0.7081 | 0.7070 | 0.7077 |
| R1 | 3 | ucr85:Gun_Point | 1/1 | 0.8751 | 0.8750 | 0.8754 |
| R1 | 3 | ucr85:Ham | 1/1 | 0.6406 | 0.6287 | 0.6432 |
| R1 | 3 | ucr85:HandOutlines | 1/1 | 0.8078 | 0.8150 | 0.8333 |
| R1 | 3 | ucr85:Haptics | 1/1 | 0.4091 | 0.3816 | 0.4136 |
| R1 | 3 | ucr85:Herring | 1/1 | 0.5341 | 0.4564 | 0.5793 |
| R1 | 3 | ucr85:InlineSkate | 1/1 | 0.3227 | 0.3122 | 0.3220 |
| R1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5596 | 0.5470 | 0.5596 |
| R1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9438 | 0.9437 | 0.9438 |
| R1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.6355 | 0.6345 | 0.6355 |
| R1 | 3 | ucr85:Lighting2 | 1/1 | 0.6835 | 0.6798 | 0.6961 |
| R1 | 3 | ucr85:Lighting7 | 1/1 | 0.5971 | 0.5567 | 0.6112 |
| R1 | 3 | ucr85:MALLAT | 1/1 | 0.7486 | 0.7116 | 0.7468 |
| R1 | 3 | ucr85:MedicalImages | 1/1 | 0.4403 | 0.4502 | 0.6326 |
| R1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.5658 | 0.5076 | 0.6673 |
| R1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5909 | 0.5430 | 0.5454 |
| R1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3722 | 0.3342 | 0.6138 |
| R1 | 3 | ucr85:MoteStrain | 1/1 | 0.7280 | 0.6973 | 0.7322 |
| R1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.8561 | 0.8461 | 0.8572 |
| R1 | 3 | ucr85:OSULeaf | 1/1 | 0.6546 | 0.6384 | 0.6653 |
| R1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7146 | 0.7080 | 0.7441 |
| R1 | 3 | ucr85:Plane | 1/1 | 0.9799 | 0.9790 | 0.9795 |
| R1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7745 | 0.7549 | 0.8435 |
| R1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8222 | 0.8333 | 0.8644 |
| R1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4590 | 0.4378 | 0.7942 |
| R1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.4550 | 0.4486 | 0.4550 |
| R1 | 3 | ucr85:ScreenType | 1/1 | 0.4622 | 0.4586 | 0.4622 |
| R1 | 3 | ucr85:ShapeletSim | 1/1 | 0.6295 | 0.6054 | 0.6295 |
| R1 | 3 | ucr85:ShapesAll | 1/1 | 0.6873 | 0.6626 | 0.6873 |
| R1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.6111 | 0.6033 | 0.6111 |
| R1 | 3 | ucr85:StarLightCurves | 1/1 | 0.8961 | 0.8900 | 0.9322 |
| R1 | 3 | ucr85:Strawberry | 1/1 | 0.9493 | 0.9457 | 0.9498 |
| R1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.8592 | 0.8500 | 0.8553 |
| R1 | 3 | ucr85:Symbols | 1/1 | 0.7441 | 0.7249 | 0.7440 |
| R1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.7368 | 0.7345 | 0.7348 |
| R1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.6970 | 0.6404 | 0.7000 |
| R1 | 3 | ucr85:Trace | 1/1 | 0.8826 | 0.8606 | 0.8723 |
| R1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.7369 | 0.7196 | 0.7368 |
| R1 | 3 | ucr85:Two_Patterns | 1/1 | 0.8992 | 0.8987 | 0.8991 |
| R1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.2372 | 0.2160 | 0.4486 |
| R1 | 3 | ucr85:Worms | 1/1 | 0.3432 | 0.3245 | 0.4441 |
| R1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.5895 | 0.5791 | 0.6235 |
| R1 | 3 | ucr85:synthetic_control | 1/1 | 0.9482 | 0.9456 | 0.9482 |
| R1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7315 | 0.7192 | 0.7311 |
| R1 | 3 | ucr85:wafer | 1/1 | 0.6596 | 0.6766 | 0.9230 |
| R1 | 3 | ucr85:yoga | 1/1 | 0.7109 | 0.7099 | 0.7162 |
| A0 | 1 | biology5:ALLAML | 1/1 | 0.9800 | 0.9845 | 0.9861 |
| A0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.8399 | 0.8399 | 0.7838 |
| A0 | 1 | biology5:Carcinom | 1/1 | 0.8297 | 0.8475 | 0.8793 |
| A0 | 1 | biology5:GLIOMA | 1/1 | 0.7155 | 0.7191 | 0.7800 |
| A0 | 1 | biology5:colon | 1/1 | 0.7102 | 0.7195 | 0.7581 |
| A0 | 1 | raman34:cancer_cell_media | 3/3 | 0.9477 | 0.9475 | 0.9472 |
| A0 | 1 | raman34:chlorinated_samples | 1/1 | 0.9180 | 0.9250 | 0.9348 |
| A0 | 1 | raman34:comfile_stroke | 1/1 | 0.8500 | 0.8496 | 0.8500 |
| A0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| A0 | 1 | raman34:diabetes_skin | 4/4 | 0.5000 | 0.3548 | 0.5500 |
| A0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.8484 | 0.8755 | 0.8943 |
| A0 | 1 | raman34:head_neck_cancer | 1/1 | 0.5045 | 0.5014 | 0.5946 |
| A0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.7700 | 0.7700 | 0.7700 |
| A0 | 1 | raman34:marine_pathogens | 2/2 | 0.9747 | 0.9754 | 0.9760 |
| A0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.9823 | 0.9824 | 0.9829 |
| A0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.3238 | 0.3188 | 0.3529 |
| A0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.7806 | 0.7806 | 0.7831 |
| A0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8000 | 0.7997 | 0.8000 |
| A0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.8820 | 0.9054 | 0.9300 |
| A0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.5212 | 0.4725 | 0.6905 |
| A0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8244 | 0.8253 | 0.8269 |
| A0 | 1 | tabmini25:backache | 1/1 | 0.5000 | 0.4627 | 0.8611 |
| A0 | 1 | tabmini25:biomed | 1/1 | 0.8376 | 0.8523 | 0.8708 |
| A0 | 1 | tabmini25:breast_cancer | 1/1 | 0.6073 | 0.6132 | 0.7343 |
| A0 | 1 | tabmini25:bupa | 1/1 | 0.5751 | 0.5730 | 0.5739 |
| A0 | 1 | tabmini25:corral | 1/1 | 0.8897 | 0.8915 | 0.8938 |
| A0 | 1 | tabmini25:haberman | 1/1 | 0.5553 | 0.5479 | 0.7353 |
| A0 | 1 | tabmini25:hepatitis | 1/1 | 0.6359 | 0.6614 | 0.8258 |
| A0 | 1 | tabmini25:irish | 1/1 | 0.9914 | 0.9919 | 0.9920 |
| A0 | 1 | tabmini25:labor | 1/1 | 0.9115 | 0.9211 | 0.9298 |
| A0 | 1 | tabmini25:lupus | 1/1 | 0.6992 | 0.7034 | 0.7241 |
| A0 | 1 | tabmini25:molecular_biology_promoters | 1/1 | 0.8679 | 0.8679 | 0.8679 |
| A0 | 1 | tabmini25:mux6 | 1/1 | 0.5938 | 0.5938 | 0.5938 |
| A0 | 1 | tabmini25:postoperative_patient_data | 1/1 | 0.5000 | 0.4211 | 0.7273 |
| A0 | 1 | tabmini25:prnn_synth | 1/1 | 0.8440 | 0.8440 | 0.8440 |
| A0 | 1 | tabmini25:saheart | 1/1 | 0.6534 | 0.6599 | 0.7121 |
| A0 | 1 | ucr85:Adiac | 1/1 | 0.5921 | 0.5613 | 0.5754 |
| A0 | 1 | ucr85:ArrowHead | 1/1 | 0.6151 | 0.6138 | 0.6171 |
| A0 | 1 | ucr85:CBF | 1/1 | 0.8528 | 0.8514 | 0.8522 |
| A0 | 1 | ucr85:Car | 1/1 | 0.6736 | 0.6768 | 0.6667 |
| A0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.4969 | 0.4986 | 0.6372 |
| A0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.6164 | 0.5952 | 0.6145 |
| A0 | 1 | ucr85:Computers | 1/1 | 0.6000 | 0.5999 | 0.6000 |
| A0 | 1 | ucr85:Cricket | 3/3 | 0.4790 | 0.4729 | 0.4744 |
| A0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.7287 | 0.7661 | 0.8500 |
| A0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.8118 | 0.8121 | 0.8250 |
| A0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4864 | 0.4586 | 0.7825 |
| A0 | 1 | ucr85:ECG200 | 1/1 | 0.7726 | 0.7835 | 0.8100 |
| A0 | 1 | ucr85:ECG5000 | 1/1 | 0.5169 | 0.5498 | 0.9367 |
| A0 | 1 | ucr85:Earthquakes | 1/1 | 0.4937 | 0.4749 | 0.7764 |
| A0 | 1 | ucr85:ElectricDevices | 1/1 | 0.4886 | 0.4836 | 0.6102 |
| A0 | 1 | ucr85:FISH | 1/1 | 0.7067 | 0.6954 | 0.7029 |
| A0 | 1 | ucr85:FaceAll | 1/1 | 0.7768 | 0.7047 | 0.6959 |
| A0 | 1 | ucr85:FaceFour | 1/1 | 0.5962 | 0.4917 | 0.5455 |
| A0 | 1 | ucr85:FacesUCR | 1/1 | 0.5957 | 0.6082 | 0.6839 |
| A0 | 1 | ucr85:FordA | 1/1 | 0.5856 | 0.5854 | 0.5865 |
| A0 | 1 | ucr85:FordB | 1/1 | 0.5352 | 0.5335 | 0.5341 |
| A0 | 1 | ucr85:Gun_Point | 1/1 | 0.8588 | 0.8586 | 0.8600 |
| A0 | 1 | ucr85:Ham | 1/1 | 0.7326 | 0.7327 | 0.7333 |
| A0 | 1 | ucr85:HandOutlines | 1/1 | 0.8019 | 0.8169 | 0.8410 |
| A0 | 1 | ucr85:Haptics | 1/1 | 0.4624 | 0.4268 | 0.4643 |
| A0 | 1 | ucr85:Herring | 1/1 | 0.5172 | 0.4868 | 0.5781 |
| A0 | 1 | ucr85:InlineSkate | 1/1 | 0.3108 | 0.3054 | 0.3073 |
| A0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.6364 | 0.6372 | 0.6364 |
| A0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9641 | 0.9640 | 0.9640 |
| A0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.5920 | 0.5912 | 0.5920 |
| A0 | 1 | ucr85:Lighting2 | 1/1 | 0.6055 | 0.5951 | 0.6230 |
| A0 | 1 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| A0 | 1 | ucr85:MALLAT | 1/1 | 0.7529 | 0.7157 | 0.7510 |
| A0 | 1 | ucr85:MedicalImages | 1/1 | 0.4586 | 0.5046 | 0.6434 |
| A0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6687 | 0.6658 | 0.7800 |
| A0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.6080 | 0.5646 | 0.5650 |
| A0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3890 | 0.3584 | 0.6366 |
| A0 | 1 | ucr85:MoteStrain | 1/1 | 0.5000 | 0.3155 | 0.4609 |
| A0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.8734 | 0.8705 | 0.8748 |
| A0 | 1 | ucr85:OSULeaf | 1/1 | 0.4413 | 0.4143 | 0.4463 |
| A0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7192 | 0.7292 | 0.7681 |
| A0 | 1 | ucr85:Plane | 1/1 | 0.9726 | 0.9724 | 0.9714 |
| A0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8017 | 0.7890 | 0.8634 |
| A0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7847 | 0.8053 | 0.8454 |
| A0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.4447 | 0.4063 | 0.8100 |
| A0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.4987 | 0.4791 | 0.4987 |
| A0 | 1 | ucr85:ScreenType | 1/1 | 0.4320 | 0.4233 | 0.4320 |
| A0 | 1 | ucr85:ShapeletSim | 1/1 | 0.5000 | 0.3333 | 0.5000 |
| A0 | 1 | ucr85:ShapesAll | 1/1 | 0.6083 | 0.5829 | 0.6083 |
| A0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.6933 | 0.6750 | 0.6933 |
| A0 | 1 | ucr85:StarLightCurves | 1/1 | 0.8932 | 0.9025 | 0.9339 |
| A0 | 1 | ucr85:Strawberry | 1/1 | 0.9416 | 0.9399 | 0.9445 |
| A0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.8307 | 0.8228 | 0.8240 |
| A0 | 1 | ucr85:Symbols | 1/1 | 0.3699 | 0.2884 | 0.3678 |
| A0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.5542 | 0.5526 | 0.5526 |
| A0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.6769 | 0.5779 | 0.6308 |
| A0 | 1 | ucr85:Trace | 1/1 | 0.8708 | 0.8490 | 0.8600 |
| A0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.5000 | 0.3331 | 0.4996 |
| A0 | 1 | ucr85:Two_Patterns | 1/1 | 0.6030 | 0.6025 | 0.6030 |
| A0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.2588 | 0.2567 | 0.4796 |
| A0 | 1 | ucr85:Worms | 1/1 | 0.3078 | 0.3020 | 0.4088 |
| A0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5685 | 0.5519 | 0.6133 |
| A0 | 1 | ucr85:synthetic_control | 1/1 | 0.9133 | 0.9129 | 0.9133 |
| A0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.7337 | 0.7251 | 0.7336 |
| A0 | 1 | ucr85:wafer | 1/1 | 0.9251 | 0.9445 | 0.9796 |
| A0 | 1 | ucr85:yoga | 1/1 | 0.7463 | 0.7472 | 0.7500 |
| A1 | 3 | biology5:ALLAML | 1/1 | 0.9600 | 0.9688 | 0.9722 |
| A1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.8530 | 0.8531 | 0.8018 |
| A1 | 3 | biology5:Carcinom | 1/1 | 0.7852 | 0.8118 | 0.8621 |
| A1 | 3 | biology5:GLIOMA | 1/1 | 0.7155 | 0.7191 | 0.7800 |
| A1 | 3 | biology5:colon | 1/1 | 0.7102 | 0.7195 | 0.7581 |
| A1 | 3 | raman34:cancer_cell_media | 3/3 | 0.9440 | 0.9440 | 0.9435 |
| A1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9180 | 0.9250 | 0.9348 |
| A1 | 3 | raman34:comfile_stroke | 1/1 | 0.8500 | 0.8496 | 0.8500 |
| A1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| A1 | 3 | raman34:diabetes_skin | 4/4 | 0.5000 | 0.3548 | 0.5500 |
| A1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9399 | 0.9500 | 0.9510 |
| A1 | 3 | raman34:head_neck_cancer | 1/1 | 0.5045 | 0.5014 | 0.5946 |
| A1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.7850 | 0.7850 | 0.7850 |
| A1 | 3 | raman34:marine_pathogens | 2/2 | 0.9723 | 0.9734 | 0.9741 |
| A1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9829 | 0.9829 | 0.9835 |
| A1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.3238 | 0.3188 | 0.3529 |
| A1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.7914 | 0.7922 | 0.7952 |
| A1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8000 | 0.7997 | 0.8000 |
| A1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.8196 | 0.8470 | 0.8900 |
| A1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5000 | 0.4085 | 0.6905 |
| A1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8244 | 0.8253 | 0.8269 |
| A1 | 3 | tabmini25:backache | 1/1 | 0.5000 | 0.4627 | 0.8611 |
| A1 | 3 | tabmini25:biomed | 1/1 | 0.8480 | 0.8632 | 0.8804 |
| A1 | 3 | tabmini25:breast_cancer | 1/1 | 0.6048 | 0.6103 | 0.7308 |
| A1 | 3 | tabmini25:bupa | 1/1 | 0.5513 | 0.5506 | 0.5507 |
| A1 | 3 | tabmini25:corral | 1/1 | 0.8571 | 0.8593 | 0.8625 |
| A1 | 3 | tabmini25:haberman | 1/1 | 0.5390 | 0.5229 | 0.7288 |
| A1 | 3 | tabmini25:hepatitis | 1/1 | 0.6203 | 0.6426 | 0.8194 |
| A1 | 3 | tabmini25:irish | 1/1 | 0.9932 | 0.9939 | 0.9940 |
| A1 | 3 | tabmini25:labor | 1/1 | 0.8115 | 0.8324 | 0.8596 |
| A1 | 3 | tabmini25:lupus | 1/1 | 0.6896 | 0.6930 | 0.7126 |
| A1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.8491 | 0.8490 | 0.8491 |
| A1 | 3 | tabmini25:mux6 | 1/1 | 0.5781 | 0.5780 | 0.5781 |
| A1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.5000 | 0.4211 | 0.7273 |
| A1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8440 | 0.8440 | 0.8440 |
| A1 | 3 | tabmini25:saheart | 1/1 | 0.6470 | 0.6529 | 0.7056 |
| A1 | 3 | ucr85:Adiac | 1/1 | 0.5918 | 0.5640 | 0.5780 |
| A1 | 3 | ucr85:ArrowHead | 1/1 | 0.6151 | 0.6138 | 0.6171 |
| A1 | 3 | ucr85:CBF | 1/1 | 0.8528 | 0.8514 | 0.8522 |
| A1 | 3 | ucr85:Car | 1/1 | 0.6736 | 0.6768 | 0.6667 |
| A1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.6134 | 0.6415 | 0.7169 |
| A1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.6164 | 0.5952 | 0.6145 |
| A1 | 3 | ucr85:Computers | 1/1 | 0.6000 | 0.5999 | 0.6000 |
| A1 | 3 | ucr85:Cricket | 3/3 | 0.5028 | 0.4978 | 0.4983 |
| A1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.7163 | 0.7568 | 0.8450 |
| A1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.8118 | 0.8121 | 0.8250 |
| A1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4864 | 0.4586 | 0.7825 |
| A1 | 3 | ucr85:ECG200 | 1/1 | 0.7726 | 0.7835 | 0.8100 |
| A1 | 3 | ucr85:ECG5000 | 1/1 | 0.5169 | 0.5498 | 0.9367 |
| A1 | 3 | ucr85:Earthquakes | 1/1 | 0.4937 | 0.4749 | 0.7764 |
| A1 | 3 | ucr85:ElectricDevices | 1/1 | 0.5624 | 0.5714 | 0.6650 |
| A1 | 3 | ucr85:FISH | 1/1 | 0.7067 | 0.6954 | 0.7029 |
| A1 | 3 | ucr85:FaceAll | 1/1 | 0.7851 | 0.7169 | 0.7065 |
| A1 | 3 | ucr85:FaceFour | 1/1 | 0.5962 | 0.4917 | 0.5455 |
| A1 | 3 | ucr85:FacesUCR | 1/1 | 0.6084 | 0.6206 | 0.6951 |
| A1 | 3 | ucr85:FordA | 1/1 | 0.6776 | 0.6776 | 0.6779 |
| A1 | 3 | ucr85:FordB | 1/1 | 0.5505 | 0.5484 | 0.5492 |
| A1 | 3 | ucr85:Gun_Point | 1/1 | 0.8588 | 0.8586 | 0.8600 |
| A1 | 3 | ucr85:Ham | 1/1 | 0.7326 | 0.7327 | 0.7333 |
| A1 | 3 | ucr85:HandOutlines | 1/1 | 0.8019 | 0.8169 | 0.8410 |
| A1 | 3 | ucr85:Haptics | 1/1 | 0.4624 | 0.4268 | 0.4643 |
| A1 | 3 | ucr85:Herring | 1/1 | 0.5172 | 0.4868 | 0.5781 |
| A1 | 3 | ucr85:InlineSkate | 1/1 | 0.3254 | 0.3256 | 0.3145 |
| A1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.6379 | 0.6377 | 0.6379 |
| A1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9533 | 0.9533 | 0.9534 |
| A1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.6053 | 0.6049 | 0.6053 |
| A1 | 3 | ucr85:Lighting2 | 1/1 | 0.6055 | 0.5951 | 0.6230 |
| A1 | 3 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| A1 | 3 | ucr85:MALLAT | 1/1 | 0.7529 | 0.7157 | 0.7510 |
| A1 | 3 | ucr85:MedicalImages | 1/1 | 0.4917 | 0.5294 | 0.6697 |
| A1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6781 | 0.6766 | 0.7900 |
| A1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.6308 | 0.5974 | 0.6000 |
| A1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3890 | 0.3584 | 0.6366 |
| A1 | 3 | ucr85:MoteStrain | 1/1 | 0.5000 | 0.3155 | 0.4609 |
| A1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.8767 | 0.8736 | 0.8784 |
| A1 | 3 | ucr85:OSULeaf | 1/1 | 0.4621 | 0.4392 | 0.4711 |
| A1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7786 | 0.7885 | 0.8089 |
| A1 | 3 | ucr85:Plane | 1/1 | 0.9726 | 0.9724 | 0.9714 |
| A1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8017 | 0.7890 | 0.8634 |
| A1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8148 | 0.8314 | 0.8625 |
| A1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4813 | 0.4603 | 0.8200 |
| A1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.5120 | 0.4959 | 0.5120 |
| A1 | 3 | ucr85:ScreenType | 1/1 | 0.4000 | 0.3943 | 0.4000 |
| A1 | 3 | ucr85:ShapeletSim | 1/1 | 0.5000 | 0.3333 | 0.5000 |
| A1 | 3 | ucr85:ShapesAll | 1/1 | 0.6083 | 0.5829 | 0.6083 |
| A1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.6933 | 0.6750 | 0.6933 |
| A1 | 3 | ucr85:StarLightCurves | 1/1 | 0.9110 | 0.9191 | 0.9446 |
| A1 | 3 | ucr85:Strawberry | 1/1 | 0.9416 | 0.9399 | 0.9445 |
| A1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.8417 | 0.8347 | 0.8352 |
| A1 | 3 | ucr85:Symbols | 1/1 | 0.3699 | 0.2884 | 0.3678 |
| A1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.5542 | 0.5526 | 0.5526 |
| A1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.6769 | 0.5779 | 0.6308 |
| A1 | 3 | ucr85:Trace | 1/1 | 0.8708 | 0.8490 | 0.8600 |
| A1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.5000 | 0.3331 | 0.4996 |
| A1 | 3 | ucr85:Two_Patterns | 1/1 | 0.7663 | 0.7657 | 0.7660 |
| A1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.2588 | 0.2567 | 0.4796 |
| A1 | 3 | ucr85:Worms | 1/1 | 0.3078 | 0.3020 | 0.4088 |
| A1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.5685 | 0.5519 | 0.6133 |
| A1 | 3 | ucr85:synthetic_control | 1/1 | 0.9133 | 0.9132 | 0.9133 |
| A1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7643 | 0.7568 | 0.7642 |
| A1 | 3 | ucr85:wafer | 1/1 | 0.9251 | 0.9445 | 0.9796 |
| A1 | 3 | ucr85:yoga | 1/1 | 0.7698 | 0.7709 | 0.7733 |
| F0 | 1 | biology5:ALLAML | 1/1 | 0.9587 | 0.9545 | 0.9583 |
| F0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.8397 | 0.8397 | 0.7838 |
| F0 | 1 | biology5:Carcinom | 1/1 | 0.8796 | 0.8945 | 0.9195 |
| F0 | 1 | biology5:GLIOMA | 1/1 | 0.7179 | 0.7260 | 0.7600 |
| F0 | 1 | biology5:colon | 1/1 | 0.8364 | 0.8398 | 0.8548 |
| F0 | 1 | raman34:cancer_cell_media | 3/3 | 0.9416 | 0.9414 | 0.9419 |
| F0 | 1 | raman34:chlorinated_samples | 1/1 | 0.9212 | 0.9298 | 0.9391 |
| F0 | 1 | raman34:comfile_stroke | 1/1 | 0.9000 | 0.9000 | 0.9000 |
| F0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 0.9333 | 0.9327 | 0.9333 |
| F0 | 1 | raman34:diabetes_skin | 4/4 | 0.5227 | 0.4916 | 0.5500 |
| F0 | 1 | raman34:hair_dyes_sers | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| F0 | 1 | raman34:head_neck_cancer | 1/1 | 0.5254 | 0.4759 | 0.6486 |
| F0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.9300 | 0.9300 | 0.9300 |
| F0 | 1 | raman34:marine_pathogens | 2/2 | 0.9890 | 0.9891 | 0.9895 |
| F0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.9732 | 0.9750 | 0.9729 |
| F0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.2571 | 0.2222 | 0.2941 |
| F0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.7832 | 0.7816 | 0.7831 |
| F0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8200 | 0.8199 | 0.8200 |
| F0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.8059 | 0.8243 | 0.8700 |
| F0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.5637 | 0.5579 | 0.6905 |
| F0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8274 | 0.8269 | 0.8269 |
| F0 | 1 | tabmini25:backache | 1/1 | 0.4968 | 0.4611 | 0.8556 |
| F0 | 1 | tabmini25:biomed | 1/1 | 0.8376 | 0.8523 | 0.8708 |
| F0 | 1 | tabmini25:breast_cancer | 1/1 | 0.5156 | 0.4843 | 0.6818 |
| F0 | 1 | tabmini25:bupa | 1/1 | 0.6363 | 0.6334 | 0.6348 |
| F0 | 1 | tabmini25:corral | 1/1 | 0.9071 | 0.9052 | 0.9062 |
| F0 | 1 | tabmini25:haberman | 1/1 | 0.5057 | 0.4453 | 0.7320 |
| F0 | 1 | tabmini25:hepatitis | 1/1 | 0.6047 | 0.6231 | 0.8129 |
| F0 | 1 | tabmini25:irish | 1/1 | 0.7529 | 0.7546 | 0.7600 |
| F0 | 1 | tabmini25:labor | 1/1 | 0.8095 | 0.8199 | 0.8421 |
| F0 | 1 | tabmini25:lupus | 1/1 | 0.7516 | 0.7559 | 0.7701 |
| F0 | 1 | tabmini25:molecular_biology_promoters | 1/1 | 0.7736 | 0.7733 | 0.7736 |
| F0 | 1 | tabmini25:mux6 | 1/1 | 0.5625 | 0.5586 | 0.5625 |
| F0 | 1 | tabmini25:postoperative_patient_data | 1/1 | 0.5000 | 0.4211 | 0.7273 |
| F0 | 1 | tabmini25:prnn_synth | 1/1 | 0.8440 | 0.8440 | 0.8440 |
| F0 | 1 | tabmini25:saheart | 1/1 | 0.6567 | 0.6637 | 0.7165 |
| F0 | 1 | ucr85:Adiac | 1/1 | 0.6196 | 0.5589 | 0.6036 |
| F0 | 1 | ucr85:ArrowHead | 1/1 | 0.7346 | 0.7268 | 0.7257 |
| F0 | 1 | ucr85:CBF | 1/1 | 0.8917 | 0.8909 | 0.8911 |
| F0 | 1 | ucr85:Car | 1/1 | 0.8526 | 0.8515 | 0.8500 |
| F0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.7680 | 0.7903 | 0.8135 |
| F0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.4154 | 0.4144 | 0.4145 |
| F0 | 1 | ucr85:Computers | 1/1 | 0.5080 | 0.5051 | 0.5080 |
| F0 | 1 | ucr85:Cricket | 3/3 | 0.3850 | 0.3670 | 0.3778 |
| F0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5183 | 0.4967 | 0.7600 |
| F0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.6986 | 0.7073 | 0.7467 |
| F0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4629 | 0.4288 | 0.7725 |
| F0 | 1 | ucr85:ECG200 | 1/1 | 0.7691 | 0.7707 | 0.7900 |
| F0 | 1 | ucr85:ECG5000 | 1/1 | 0.4227 | 0.4255 | 0.9258 |
| F0 | 1 | ucr85:Earthquakes | 1/1 | 0.5000 | 0.4505 | 0.8199 |
| F0 | 1 | ucr85:ElectricDevices | 1/1 | 0.3680 | 0.3424 | 0.4911 |
| F0 | 1 | ucr85:FISH | 1/1 | 0.8541 | 0.8503 | 0.8514 |
| F0 | 1 | ucr85:FaceAll | 1/1 | 0.7973 | 0.7340 | 0.7118 |
| F0 | 1 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| F0 | 1 | ucr85:FacesUCR | 1/1 | 0.6417 | 0.6448 | 0.7244 |
| F0 | 1 | ucr85:FordA | 1/1 | 0.5000 | 0.3389 | 0.5126 |
| F0 | 1 | ucr85:FordB | 1/1 | 0.4990 | 0.4986 | 0.4986 |
| F0 | 1 | ucr85:Gun_Point | 1/1 | 0.8994 | 0.8996 | 0.9000 |
| F0 | 1 | ucr85:Ham | 1/1 | 0.6187 | 0.6187 | 0.6190 |
| F0 | 1 | ucr85:HandOutlines | 1/1 | 0.7983 | 0.8054 | 0.8250 |
| F0 | 1 | ucr85:Haptics | 1/1 | 0.4134 | 0.3802 | 0.4188 |
| F0 | 1 | ucr85:Herring | 1/1 | 0.5000 | 0.3725 | 0.5938 |
| F0 | 1 | ucr85:InlineSkate | 1/1 | 0.2151 | 0.1809 | 0.2364 |
| F0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.5944 | 0.5912 | 0.5944 |
| F0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9611 | 0.9611 | 0.9611 |
| F0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.3867 | 0.3861 | 0.3867 |
| F0 | 1 | ucr85:Lighting2 | 1/1 | 0.6634 | 0.6420 | 0.6885 |
| F0 | 1 | ucr85:Lighting7 | 1/1 | 0.5937 | 0.5534 | 0.6164 |
| F0 | 1 | ucr85:MALLAT | 1/1 | 0.8617 | 0.8559 | 0.8606 |
| F0 | 1 | ucr85:MedicalImages | 1/1 | 0.3825 | 0.4273 | 0.6118 |
| F0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.4174 | 0.2978 | 0.4225 |
| F0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5739 | 0.5329 | 0.5333 |
| F0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3368 | 0.2810 | 0.6040 |
| F0 | 1 | ucr85:MoteStrain | 1/1 | 0.8685 | 0.8683 | 0.8690 |
| F0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9296 | 0.9280 | 0.9310 |
| F0 | 1 | ucr85:OSULeaf | 1/1 | 0.3812 | 0.3432 | 0.3967 |
| F0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.5872 | 0.5659 | 0.6655 |
| F0 | 1 | ucr85:Plane | 1/1 | 0.9911 | 0.9908 | 0.9905 |
| F0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7235 | 0.7151 | 0.8244 |
| F0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7980 | 0.8190 | 0.8557 |
| F0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.4866 | 0.4680 | 0.8175 |
| F0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.3600 | 0.3409 | 0.3600 |
| F0 | 1 | ucr85:ScreenType | 1/1 | 0.3733 | 0.3705 | 0.3733 |
| F0 | 1 | ucr85:ShapeletSim | 1/1 | 0.5389 | 0.5052 | 0.5389 |
| F0 | 1 | ucr85:ShapesAll | 1/1 | 0.6667 | 0.6307 | 0.6667 |
| F0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.4640 | 0.4586 | 0.4640 |
| F0 | 1 | ucr85:StarLightCurves | 1/1 | 0.8842 | 0.8886 | 0.9246 |
| F0 | 1 | ucr85:Strawberry | 1/1 | 0.9650 | 0.9612 | 0.9641 |
| F0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.8706 | 0.8680 | 0.8688 |
| F0 | 1 | ucr85:Symbols | 1/1 | 0.6849 | 0.6799 | 0.6884 |
| F0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.5588 | 0.5570 | 0.5570 |
| F0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.3797 | 0.3765 | 0.4615 |
| F0 | 1 | ucr85:Trace | 1/1 | 0.8318 | 0.8118 | 0.8200 |
| F0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.9403 | 0.9403 | 0.9403 |
| F0 | 1 | ucr85:Two_Patterns | 1/1 | 0.8090 | 0.8087 | 0.8087 |
| F0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.2141 | 0.1998 | 0.4577 |
| F0 | 1 | ucr85:Worms | 1/1 | 0.2000 | 0.1183 | 0.4199 |
| F0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5148 | 0.4868 | 0.5635 |
| F0 | 1 | ucr85:synthetic_control | 1/1 | 0.9400 | 0.9399 | 0.9400 |
| F0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.6761 | 0.6650 | 0.6756 |
| F0 | 1 | ucr85:wafer | 1/1 | 0.7378 | 0.8005 | 0.9401 |
| F0 | 1 | ucr85:yoga | 1/1 | 0.6575 | 0.6571 | 0.6650 |
| F1 | 3 | biology5:ALLAML | 1/1 | 0.9587 | 0.9545 | 0.9583 |
| F1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.8397 | 0.8397 | 0.7838 |
| F1 | 3 | biology5:Carcinom | 1/1 | 0.8796 | 0.8945 | 0.9195 |
| F1 | 3 | biology5:GLIOMA | 1/1 | 0.7179 | 0.7260 | 0.7600 |
| F1 | 3 | biology5:colon | 1/1 | 0.8364 | 0.8398 | 0.8548 |
| F1 | 3 | raman34:cancer_cell_media | 3/3 | 0.9416 | 0.9414 | 0.9419 |
| F1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9212 | 0.9298 | 0.9391 |
| F1 | 3 | raman34:comfile_stroke | 1/1 | 0.9000 | 0.9000 | 0.9000 |
| F1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 0.9333 | 0.9327 | 0.9333 |
| F1 | 3 | raman34:diabetes_skin | 4/4 | 0.5227 | 0.4916 | 0.5500 |
| F1 | 3 | raman34:hair_dyes_sers | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| F1 | 3 | raman34:head_neck_cancer | 1/1 | 0.5386 | 0.4947 | 0.6577 |
| F1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.9300 | 0.9300 | 0.9300 |
| F1 | 3 | raman34:marine_pathogens | 2/2 | 0.9890 | 0.9891 | 0.9895 |
| F1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9774 | 0.9786 | 0.9775 |
| F1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2571 | 0.2222 | 0.2941 |
| F1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8102 | 0.8065 | 0.8072 |
| F1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8200 | 0.8199 | 0.8200 |
| F1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.8361 | 0.8441 | 0.8800 |
| F1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5637 | 0.5579 | 0.6905 |
| F1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8274 | 0.8269 | 0.8269 |
| F1 | 3 | tabmini25:backache | 1/1 | 0.4968 | 0.4611 | 0.8556 |
| F1 | 3 | tabmini25:biomed | 1/1 | 0.8376 | 0.8523 | 0.8708 |
| F1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5156 | 0.4843 | 0.6818 |
| F1 | 3 | tabmini25:bupa | 1/1 | 0.6364 | 0.6332 | 0.6348 |
| F1 | 3 | tabmini25:corral | 1/1 | 0.9071 | 0.9052 | 0.9062 |
| F1 | 3 | tabmini25:haberman | 1/1 | 0.5057 | 0.4453 | 0.7320 |
| F1 | 3 | tabmini25:hepatitis | 1/1 | 0.6278 | 0.6485 | 0.8129 |
| F1 | 3 | tabmini25:irish | 1/1 | 0.7529 | 0.7546 | 0.7600 |
| F1 | 3 | tabmini25:labor | 1/1 | 0.8095 | 0.8199 | 0.8421 |
| F1 | 3 | tabmini25:lupus | 1/1 | 0.7374 | 0.7421 | 0.7586 |
| F1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.7736 | 0.7733 | 0.7736 |
| F1 | 3 | tabmini25:mux6 | 1/1 | 0.6094 | 0.6085 | 0.6094 |
| F1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.5000 | 0.4211 | 0.7273 |
| F1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8480 | 0.8480 | 0.8480 |
| F1 | 3 | tabmini25:saheart | 1/1 | 0.6567 | 0.6637 | 0.7165 |
| F1 | 3 | ucr85:Adiac | 1/1 | 0.6196 | 0.5589 | 0.6036 |
| F1 | 3 | ucr85:ArrowHead | 1/1 | 0.7346 | 0.7268 | 0.7257 |
| F1 | 3 | ucr85:CBF | 1/1 | 0.8917 | 0.8909 | 0.8911 |
| F1 | 3 | ucr85:Car | 1/1 | 0.8526 | 0.8515 | 0.8500 |
| F1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.8666 | 0.8852 | 0.8953 |
| F1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.4154 | 0.4144 | 0.4145 |
| F1 | 3 | ucr85:Computers | 1/1 | 0.4960 | 0.4939 | 0.4960 |
| F1 | 3 | ucr85:Cricket | 3/3 | 0.3836 | 0.3651 | 0.3761 |
| F1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5183 | 0.4967 | 0.7600 |
| F1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7072 | 0.7159 | 0.7517 |
| F1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4629 | 0.4288 | 0.7725 |
| F1 | 3 | ucr85:ECG200 | 1/1 | 0.7491 | 0.7549 | 0.7800 |
| F1 | 3 | ucr85:ECG5000 | 1/1 | 0.4227 | 0.4255 | 0.9258 |
| F1 | 3 | ucr85:Earthquakes | 1/1 | 0.5000 | 0.4505 | 0.8199 |
| F1 | 3 | ucr85:ElectricDevices | 1/1 | 0.3708 | 0.3469 | 0.4918 |
| F1 | 3 | ucr85:FISH | 1/1 | 0.8541 | 0.8503 | 0.8514 |
| F1 | 3 | ucr85:FaceAll | 1/1 | 0.7973 | 0.7340 | 0.7118 |
| F1 | 3 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| F1 | 3 | ucr85:FacesUCR | 1/1 | 0.6417 | 0.6448 | 0.7244 |
| F1 | 3 | ucr85:FordA | 1/1 | 0.4825 | 0.3343 | 0.4946 |
| F1 | 3 | ucr85:FordB | 1/1 | 0.5008 | 0.5006 | 0.5006 |
| F1 | 3 | ucr85:Gun_Point | 1/1 | 0.8994 | 0.8996 | 0.9000 |
| F1 | 3 | ucr85:Ham | 1/1 | 0.6187 | 0.6187 | 0.6190 |
| F1 | 3 | ucr85:HandOutlines | 1/1 | 0.8266 | 0.8372 | 0.8550 |
| F1 | 3 | ucr85:Haptics | 1/1 | 0.4397 | 0.4055 | 0.4448 |
| F1 | 3 | ucr85:Herring | 1/1 | 0.5000 | 0.3725 | 0.5938 |
| F1 | 3 | ucr85:InlineSkate | 1/1 | 0.2151 | 0.1809 | 0.2364 |
| F1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5990 | 0.5903 | 0.5990 |
| F1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9611 | 0.9611 | 0.9611 |
| F1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.3867 | 0.3861 | 0.3867 |
| F1 | 3 | ucr85:Lighting2 | 1/1 | 0.6634 | 0.6420 | 0.6885 |
| F1 | 3 | ucr85:Lighting7 | 1/1 | 0.5937 | 0.5534 | 0.6164 |
| F1 | 3 | ucr85:MALLAT | 1/1 | 0.8617 | 0.8559 | 0.8606 |
| F1 | 3 | ucr85:MedicalImages | 1/1 | 0.3825 | 0.4273 | 0.6118 |
| F1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.4174 | 0.2978 | 0.4225 |
| F1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5951 | 0.5482 | 0.5483 |
| F1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3368 | 0.2810 | 0.6040 |
| F1 | 3 | ucr85:MoteStrain | 1/1 | 0.8685 | 0.8683 | 0.8690 |
| F1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9296 | 0.9280 | 0.9310 |
| F1 | 3 | ucr85:OSULeaf | 1/1 | 0.3952 | 0.3520 | 0.4091 |
| F1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.5885 | 0.5699 | 0.6643 |
| F1 | 3 | ucr85:Plane | 1/1 | 0.9911 | 0.9908 | 0.9905 |
| F1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7904 | 0.7705 | 0.8488 |
| F1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7980 | 0.8190 | 0.8557 |
| F1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4866 | 0.4680 | 0.8175 |
| F1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.3600 | 0.3409 | 0.3600 |
| F1 | 3 | ucr85:ScreenType | 1/1 | 0.3787 | 0.3768 | 0.3787 |
| F1 | 3 | ucr85:ShapeletSim | 1/1 | 0.5389 | 0.5052 | 0.5389 |
| F1 | 3 | ucr85:ShapesAll | 1/1 | 0.6650 | 0.6276 | 0.6650 |
| F1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.4427 | 0.4396 | 0.4427 |
| F1 | 3 | ucr85:StarLightCurves | 1/1 | 0.8888 | 0.8886 | 0.9230 |
| F1 | 3 | ucr85:Strawberry | 1/1 | 0.9650 | 0.9612 | 0.9641 |
| F1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.8871 | 0.8843 | 0.8848 |
| F1 | 3 | ucr85:Symbols | 1/1 | 0.6849 | 0.6799 | 0.6884 |
| F1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.5588 | 0.5570 | 0.5570 |
| F1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.3797 | 0.3765 | 0.4615 |
| F1 | 3 | ucr85:Trace | 1/1 | 0.8318 | 0.8118 | 0.8200 |
| F1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.9403 | 0.9403 | 0.9403 |
| F1 | 3 | ucr85:Two_Patterns | 1/1 | 0.8272 | 0.8268 | 0.8267 |
| F1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.2141 | 0.1998 | 0.4577 |
| F1 | 3 | ucr85:Worms | 1/1 | 0.2000 | 0.1183 | 0.4199 |
| F1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.5148 | 0.4868 | 0.5635 |
| F1 | 3 | ucr85:synthetic_control | 1/1 | 0.9400 | 0.9399 | 0.9400 |
| F1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.6943 | 0.6822 | 0.6940 |
| F1 | 3 | ucr85:wafer | 1/1 | 0.7378 | 0.8005 | 0.9401 |
| F1 | 3 | ucr85:yoga | 1/1 | 0.6575 | 0.6571 | 0.6650 |
| P0 | 1 | biology5:ALLAML | 1/1 | 0.9600 | 0.9688 | 0.9722 |
| P0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.8195 | 0.8195 | 0.7568 |
| P0 | 1 | biology5:Carcinom | 1/1 | 0.8476 | 0.8646 | 0.8851 |
| P0 | 1 | biology5:GLIOMA | 1/1 | 0.7179 | 0.7177 | 0.7400 |
| P0 | 1 | biology5:colon | 1/1 | 0.8136 | 0.8200 | 0.8387 |
| P0 | 1 | raman34:cancer_cell_media | 3/3 | 0.9577 | 0.9577 | 0.9577 |
| P0 | 1 | raman34:chlorinated_samples | 1/1 | 0.9148 | 0.9162 | 0.9261 |
| P0 | 1 | raman34:comfile_stroke | 1/1 | 0.7000 | 0.6970 | 0.7000 |
| P0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 0.8667 | 0.8704 | 0.8667 |
| P0 | 1 | raman34:diabetes_skin | 4/4 | 0.5227 | 0.4977 | 0.5250 |
| P0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.9975 | 0.9973 | 0.9977 |
| P0 | 1 | raman34:head_neck_cancer | 1/1 | 0.4875 | 0.4739 | 0.5856 |
| P0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.8650 | 0.8650 | 0.8650 |
| P0 | 1 | raman34:marine_pathogens | 2/2 | 0.9925 | 0.9926 | 0.9929 |
| P0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.9982 | 0.9982 | 0.9983 |
| P0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.3524 | 0.2738 | 0.4118 |
| P0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.8211 | 0.8183 | 0.8193 |
| P0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8600 | 0.8599 | 0.8600 |
| P0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.9932 | 0.9875 | 0.9900 |
| P0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.6101 | 0.6101 | 0.6667 |
| P0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.7830 | 0.7819 | 0.7885 |
| P0 | 1 | tabmini25:backache | 1/1 | 0.5219 | 0.5228 | 0.7833 |
| P0 | 1 | tabmini25:biomed | 1/1 | 0.8398 | 0.8496 | 0.8660 |
| P0 | 1 | tabmini25:breast_cancer | 1/1 | 0.6057 | 0.6118 | 0.7273 |
| P0 | 1 | tabmini25:bupa | 1/1 | 0.6417 | 0.6399 | 0.6406 |
| P0 | 1 | tabmini25:corral | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| P0 | 1 | tabmini25:haberman | 1/1 | 0.5474 | 0.5344 | 0.7353 |
| P0 | 1 | tabmini25:hepatitis | 1/1 | 0.7012 | 0.7107 | 0.8194 |
| P0 | 1 | tabmini25:irish | 1/1 | 0.9906 | 0.9899 | 0.9900 |
| P0 | 1 | tabmini25:labor | 1/1 | 0.8209 | 0.8246 | 0.8421 |
| P0 | 1 | tabmini25:lupus | 1/1 | 0.7231 | 0.7281 | 0.7471 |
| P0 | 1 | tabmini25:molecular_biology_promoters | 1/1 | 0.8302 | 0.8296 | 0.8302 |
| P0 | 1 | tabmini25:mux6 | 1/1 | 0.6875 | 0.6875 | 0.6875 |
| P0 | 1 | tabmini25:postoperative_patient_data | 1/1 | 0.4974 | 0.4479 | 0.7045 |
| P0 | 1 | tabmini25:prnn_synth | 1/1 | 0.8400 | 0.8400 | 0.8400 |
| P0 | 1 | tabmini25:saheart | 1/1 | 0.6705 | 0.6771 | 0.7229 |
| P0 | 1 | ucr85:Adiac | 1/1 | 0.7472 | 0.7123 | 0.7340 |
| P0 | 1 | ucr85:ArrowHead | 1/1 | 0.5562 | 0.5340 | 0.5371 |
| P0 | 1 | ucr85:CBF | 1/1 | 0.8496 | 0.8478 | 0.8489 |
| P0 | 1 | ucr85:Car | 1/1 | 0.7629 | 0.7534 | 0.7500 |
| P0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.8630 | 0.8732 | 0.8888 |
| P0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.5980 | 0.5975 | 0.5971 |
| P0 | 1 | ucr85:Computers | 1/1 | 0.7560 | 0.7539 | 0.7560 |
| P0 | 1 | ucr85:Cricket | 3/3 | 0.7978 | 0.7928 | 0.7974 |
| P0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5585 | 0.5310 | 0.8025 |
| P0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.8118 | 0.8121 | 0.8250 |
| P0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4654 | 0.4310 | 0.7450 |
| P0 | 1 | ucr85:ECG200 | 1/1 | 0.8420 | 0.8462 | 0.8600 |
| P0 | 1 | ucr85:ECG5000 | 1/1 | 0.4694 | 0.4811 | 0.9260 |
| P0 | 1 | ucr85:Earthquakes | 1/1 | 0.5512 | 0.5393 | 0.6832 |
| P0 | 1 | ucr85:ElectricDevices | 1/1 | 0.6471 | 0.6451 | 0.7415 |
| P0 | 1 | ucr85:FISH | 1/1 | 0.8774 | 0.8749 | 0.8800 |
| P0 | 1 | ucr85:FaceAll | 1/1 | 0.9038 | 0.8413 | 0.8598 |
| P0 | 1 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| P0 | 1 | ucr85:FacesUCR | 1/1 | 0.7989 | 0.8020 | 0.8478 |
| P0 | 1 | ucr85:FordA | 1/1 | 0.9362 | 0.9363 | 0.9364 |
| P0 | 1 | ucr85:FordB | 1/1 | 0.9187 | 0.9192 | 0.9194 |
| P0 | 1 | ucr85:Gun_Point | 1/1 | 0.8535 | 0.8533 | 0.8533 |
| P0 | 1 | ucr85:Ham | 1/1 | 0.6656 | 0.6656 | 0.6667 |
| P0 | 1 | ucr85:HandOutlines | 1/1 | 0.8190 | 0.8346 | 0.8560 |
| P0 | 1 | ucr85:Haptics | 1/1 | 0.4450 | 0.4032 | 0.4481 |
| P0 | 1 | ucr85:Herring | 1/1 | 0.5020 | 0.4000 | 0.4375 |
| P0 | 1 | ucr85:InlineSkate | 1/1 | 0.3578 | 0.3171 | 0.3582 |
| P0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.6303 | 0.6282 | 0.6303 |
| P0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9320 | 0.9320 | 0.9320 |
| P0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.8427 | 0.8426 | 0.8427 |
| P0 | 1 | ucr85:Lighting2 | 1/1 | 0.6921 | 0.6909 | 0.7049 |
| P0 | 1 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| P0 | 1 | ucr85:MALLAT | 1/1 | 0.8970 | 0.8898 | 0.8968 |
| P0 | 1 | ucr85:MedicalImages | 1/1 | 0.5303 | 0.5422 | 0.6632 |
| P0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6664 | 0.6522 | 0.7950 |
| P0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5698 | 0.5093 | 0.5100 |
| P0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3751 | 0.3510 | 0.6190 |
| P0 | 1 | ucr85:MoteStrain | 1/1 | 0.7944 | 0.7948 | 0.7963 |
| P0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9248 | 0.9231 | 0.9267 |
| P0 | 1 | ucr85:OSULeaf | 1/1 | 0.9667 | 0.9713 | 0.9711 |
| P0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7563 | 0.7467 | 0.7529 |
| P0 | 1 | ucr85:Plane | 1/1 | 0.9905 | 0.9907 | 0.9905 |
| P0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8345 | 0.7784 | 0.8439 |
| P0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8607 | 0.8725 | 0.8935 |
| P0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.5161 | 0.5114 | 0.7925 |
| P0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.5547 | 0.5546 | 0.5547 |
| P0 | 1 | ucr85:ScreenType | 1/1 | 0.5093 | 0.5064 | 0.5093 |
| P0 | 1 | ucr85:ShapeletSim | 1/1 | 0.8556 | 0.8554 | 0.8556 |
| P0 | 1 | ucr85:ShapesAll | 1/1 | 0.8650 | 0.8603 | 0.8650 |
| P0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.7493 | 0.7466 | 0.7493 |
| P0 | 1 | ucr85:StarLightCurves | 1/1 | 0.9568 | 0.9494 | 0.9641 |
| P0 | 1 | ucr85:Strawberry | 1/1 | 0.9437 | 0.9401 | 0.9445 |
| P0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.9389 | 0.9380 | 0.9376 |
| P0 | 1 | ucr85:Symbols | 1/1 | 0.8102 | 0.8094 | 0.8090 |
| P0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.8870 | 0.8812 | 0.8816 |
| P0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.8090 | 0.7724 | 0.8462 |
| P0 | 1 | ucr85:Trace | 1/1 | 0.9911 | 0.9890 | 0.9900 |
| P0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.6587 | 0.6412 | 0.6585 |
| P0 | 1 | ucr85:Two_Patterns | 1/1 | 0.9975 | 0.9975 | 0.9975 |
| P0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.2745 | 0.2501 | 0.5094 |
| P0 | 1 | ucr85:Worms | 1/1 | 0.5177 | 0.5034 | 0.5746 |
| P0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.6543 | 0.6556 | 0.6685 |
| P0 | 1 | ucr85:synthetic_control | 1/1 | 0.9867 | 0.9867 | 0.9867 |
| P0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.7902 | 0.7855 | 0.7899 |
| P0 | 1 | ucr85:wafer | 1/1 | 0.6084 | 0.6535 | 0.9134 |
| P0 | 1 | ucr85:yoga | 1/1 | 0.8217 | 0.8213 | 0.8220 |
| P1 | 3 | biology5:ALLAML | 1/1 | 0.9268 | 0.9111 | 0.9167 |
| P1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.8195 | 0.8195 | 0.7568 |
| P1 | 3 | biology5:Carcinom | 1/1 | 0.8399 | 0.8540 | 0.8793 |
| P1 | 3 | biology5:GLIOMA | 1/1 | 0.7702 | 0.7797 | 0.8000 |
| P1 | 3 | biology5:colon | 1/1 | 0.8364 | 0.8398 | 0.8548 |
| P1 | 3 | raman34:cancer_cell_media | 3/3 | 0.9572 | 0.9571 | 0.9572 |
| P1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9148 | 0.9162 | 0.9261 |
| P1 | 3 | raman34:comfile_stroke | 1/1 | 0.7000 | 0.6970 | 0.7000 |
| P1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 0.8667 | 0.8704 | 0.8667 |
| P1 | 3 | raman34:diabetes_skin | 4/4 | 0.4747 | 0.4551 | 0.4750 |
| P1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9947 | 0.9949 | 0.9959 |
| P1 | 3 | raman34:head_neck_cancer | 1/1 | 0.4927 | 0.4886 | 0.5766 |
| P1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.8750 | 0.8750 | 0.8750 |
| P1 | 3 | raman34:marine_pathogens | 2/2 | 0.9919 | 0.9921 | 0.9925 |
| P1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9977 | 0.9977 | 0.9977 |
| P1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2095 | 0.1889 | 0.2353 |
| P1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8211 | 0.8183 | 0.8193 |
| P1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8600 | 0.8599 | 0.8600 |
| P1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.9932 | 0.9875 | 0.9900 |
| P1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.6101 | 0.6101 | 0.6667 |
| P1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8244 | 0.8253 | 0.8269 |
| P1 | 3 | tabmini25:backache | 1/1 | 0.5252 | 0.5264 | 0.7889 |
| P1 | 3 | tabmini25:biomed | 1/1 | 0.8398 | 0.8496 | 0.8660 |
| P1 | 3 | tabmini25:breast_cancer | 1/1 | 0.6057 | 0.6118 | 0.7273 |
| P1 | 3 | tabmini25:bupa | 1/1 | 0.5956 | 0.5930 | 0.5942 |
| P1 | 3 | tabmini25:corral | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| P1 | 3 | tabmini25:haberman | 1/1 | 0.5516 | 0.5495 | 0.7124 |
| P1 | 3 | tabmini25:hepatitis | 1/1 | 0.7284 | 0.7311 | 0.8258 |
| P1 | 3 | tabmini25:irish | 1/1 | 0.9906 | 0.9899 | 0.9900 |
| P1 | 3 | tabmini25:labor | 1/1 | 0.8209 | 0.8246 | 0.8421 |
| P1 | 3 | tabmini25:lupus | 1/1 | 0.7231 | 0.7281 | 0.7471 |
| P1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.8302 | 0.8301 | 0.8302 |
| P1 | 3 | tabmini25:mux6 | 1/1 | 0.7656 | 0.7656 | 0.7656 |
| P1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.4974 | 0.4479 | 0.7045 |
| P1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8480 | 0.8480 | 0.8480 |
| P1 | 3 | tabmini25:saheart | 1/1 | 0.6553 | 0.6623 | 0.7165 |
| P1 | 3 | ucr85:Adiac | 1/1 | 0.7404 | 0.7145 | 0.7263 |
| P1 | 3 | ucr85:ArrowHead | 1/1 | 0.7327 | 0.7216 | 0.7200 |
| P1 | 3 | ucr85:CBF | 1/1 | 0.8523 | 0.8506 | 0.8522 |
| P1 | 3 | ucr85:Car | 1/1 | 0.6887 | 0.6849 | 0.6833 |
| P1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.8459 | 0.8618 | 0.8836 |
| P1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.7328 | 0.7342 | 0.7319 |
| P1 | 3 | ucr85:Computers | 1/1 | 0.7560 | 0.7539 | 0.7560 |
| P1 | 3 | ucr85:Cricket | 3/3 | 0.7978 | 0.7928 | 0.7974 |
| P1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5805 | 0.5693 | 0.7850 |
| P1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.8118 | 0.8121 | 0.8250 |
| P1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4890 | 0.4618 | 0.7225 |
| P1 | 3 | ucr85:ECG200 | 1/1 | 0.8420 | 0.8462 | 0.8600 |
| P1 | 3 | ucr85:ECG5000 | 1/1 | 0.5219 | 0.4918 | 0.9238 |
| P1 | 3 | ucr85:Earthquakes | 1/1 | 0.6520 | 0.5394 | 0.5839 |
| P1 | 3 | ucr85:ElectricDevices | 1/1 | 0.6471 | 0.6451 | 0.7415 |
| P1 | 3 | ucr85:FISH | 1/1 | 0.8506 | 0.8482 | 0.8514 |
| P1 | 3 | ucr85:FaceAll | 1/1 | 0.9038 | 0.8413 | 0.8598 |
| P1 | 3 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| P1 | 3 | ucr85:FacesUCR | 1/1 | 0.7678 | 0.7811 | 0.8224 |
| P1 | 3 | ucr85:FordA | 1/1 | 0.9462 | 0.9458 | 0.9458 |
| P1 | 3 | ucr85:FordB | 1/1 | 0.9188 | 0.9188 | 0.9189 |
| P1 | 3 | ucr85:Gun_Point | 1/1 | 0.8535 | 0.8533 | 0.8533 |
| P1 | 3 | ucr85:Ham | 1/1 | 0.6656 | 0.6656 | 0.6667 |
| P1 | 3 | ucr85:HandOutlines | 1/1 | 0.7961 | 0.8143 | 0.8420 |
| P1 | 3 | ucr85:Haptics | 1/1 | 0.4450 | 0.4032 | 0.4481 |
| P1 | 3 | ucr85:Herring | 1/1 | 0.4737 | 0.3600 | 0.5625 |
| P1 | 3 | ucr85:InlineSkate | 1/1 | 0.4937 | 0.4967 | 0.4691 |
| P1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5965 | 0.5864 | 0.5965 |
| P1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9153 | 0.9153 | 0.9155 |
| P1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.8587 | 0.8591 | 0.8587 |
| P1 | 3 | ucr85:Lighting2 | 1/1 | 0.6136 | 0.6125 | 0.6230 |
| P1 | 3 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| P1 | 3 | ucr85:MALLAT | 1/1 | 0.8970 | 0.8898 | 0.8968 |
| P1 | 3 | ucr85:MedicalImages | 1/1 | 0.5303 | 0.5422 | 0.6632 |
| P1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6664 | 0.6522 | 0.7950 |
| P1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.6000 | 0.5359 | 0.5367 |
| P1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3751 | 0.3510 | 0.6190 |
| P1 | 3 | ucr85:MoteStrain | 1/1 | 0.8052 | 0.8025 | 0.8027 |
| P1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9232 | 0.9211 | 0.9247 |
| P1 | 3 | ucr85:OSULeaf | 1/1 | 0.9451 | 0.9510 | 0.9504 |
| P1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7563 | 0.7467 | 0.7529 |
| P1 | 3 | ucr85:Plane | 1/1 | 0.9905 | 0.9907 | 0.9905 |
| P1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7953 | 0.7544 | 0.8341 |
| P1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8674 | 0.8685 | 0.8866 |
| P1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.5157 | 0.5094 | 0.7650 |
| P1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.5547 | 0.5546 | 0.5547 |
| P1 | 3 | ucr85:ScreenType | 1/1 | 0.5093 | 0.5064 | 0.5093 |
| P1 | 3 | ucr85:ShapeletSim | 1/1 | 0.8556 | 0.8554 | 0.8556 |
| P1 | 3 | ucr85:ShapesAll | 1/1 | 0.8650 | 0.8603 | 0.8650 |
| P1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.7493 | 0.7468 | 0.7493 |
| P1 | 3 | ucr85:StarLightCurves | 1/1 | 0.9568 | 0.9494 | 0.9641 |
| P1 | 3 | ucr85:Strawberry | 1/1 | 0.9437 | 0.9401 | 0.9445 |
| P1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.9486 | 0.9475 | 0.9472 |
| P1 | 3 | ucr85:Symbols | 1/1 | 0.8102 | 0.8094 | 0.8090 |
| P1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.8870 | 0.8812 | 0.8816 |
| P1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.8090 | 0.7724 | 0.8462 |
| P1 | 3 | ucr85:Trace | 1/1 | 0.9911 | 0.9890 | 0.9900 |
| P1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.6824 | 0.6660 | 0.6822 |
| P1 | 3 | ucr85:Two_Patterns | 1/1 | 0.9975 | 0.9975 | 0.9975 |
| P1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.3120 | 0.2847 | 0.5329 |
| P1 | 3 | ucr85:Worms | 1/1 | 0.5183 | 0.5336 | 0.5801 |
| P1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.7031 | 0.7069 | 0.7293 |
| P1 | 3 | ucr85:synthetic_control | 1/1 | 0.9867 | 0.9867 | 0.9867 |
| P1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7836 | 0.7780 | 0.7832 |
| P1 | 3 | ucr85:wafer | 1/1 | 0.8100 | 0.8494 | 0.9487 |
| P1 | 3 | ucr85:yoga | 1/1 | 0.7832 | 0.7836 | 0.7850 |

## 开发任务确定的固定模型

- budget=1：模型 svm，开发任务平均BA 0.8028；候选 logistic_regression, pca_lda, pca_mlp, pls_da, random_forest, svm, xgboost。
- budget=3：模型 svm，开发任务平均BA 0.8024；候选 logistic_regression, pca_lda, pca_mlp, pls_da, random_forest, svm, xgboost。

## 配对差

差值为左侧方法减右侧方法；95% 区间按 family_id 聚类 bootstrap；tie 阈值为 1e-12。

| 比较 | 指标 | 任务数 | 平均差 | 95% CI | Win/Tie/Loss |
|---|---|---:|---:|---|---:|
| A0-R0 | balanced_accuracy | 114 | 0.0110 | [-0.0083, 0.0304] | 63/0/51 |
| A0-R0 | accuracy | 114 | 0.0079 | [-0.0102, 0.0267] | 68/1/45 |
| A0-R0 | macro_f1 | 114 | 0.0076 | [-0.0165, 0.0316] | 63/0/51 |
| A1-R1 | balanced_accuracy | 114 | -0.0022 | [-0.0187, 0.0134] | 57/0/57 |
| A1-R1 | accuracy | 114 | -0.0019 | [-0.0186, 0.0144] | 63/0/51 |
| A1-R1 | macro_f1 | 114 | -0.0075 | [-0.0289, 0.0126] | 57/0/57 |
| F0-R0 | balanced_accuracy | 114 | 0.0026 | [-0.0201, 0.0254] | 59/0/55 |
| F0-R0 | accuracy | 114 | 0.0055 | [-0.0161, 0.0273] | 65/1/48 |
| F0-R0 | macro_f1 | 114 | 0.0024 | [-0.0209, 0.0255] | 59/0/55 |
| F0-A0 | balanced_accuracy | 114 | -0.0084 | [-0.0298, 0.0136] | 49/3/62 |
| F0-A0 | accuracy | 114 | -0.0024 | [-0.0231, 0.0186] | 46/8/60 |
| F0-A0 | macro_f1 | 114 | -0.0051 | [-0.0331, 0.0214] | 48/2/64 |
| F0-P0 | balanced_accuracy | 114 | -0.0825 | [-0.1135, -0.0540] | 28/3/83 |
| F0-P0 | accuracy | 114 | -0.0686 | [-0.1006, -0.0382] | 35/5/74 |
| F0-P0 | macro_f1 | 114 | -0.0883 | [-0.1224, -0.0586] | 29/2/83 |
| F1-R1 | balanced_accuracy | 114 | -0.0125 | [-0.0321, 0.0085] | 54/0/60 |
| F1-R1 | accuracy | 114 | -0.0072 | [-0.0285, 0.0128] | 59/2/53 |
| F1-R1 | macro_f1 | 114 | -0.0152 | [-0.0375, 0.0067] | 52/0/62 |
| F1-A1 | balanced_accuracy | 114 | -0.0103 | [-0.0320, 0.0129] | 54/1/59 |
| F1-A1 | accuracy | 114 | -0.0053 | [-0.0258, 0.0164] | 50/4/60 |
| F1-A1 | macro_f1 | 114 | -0.0077 | [-0.0356, 0.0216] | 52/1/61 |
| F1-P1 | balanced_accuracy | 114 | -0.0819 | [-0.1138, -0.0521] | 34/5/75 |
| F1-P1 | accuracy | 114 | -0.0650 | [-0.0983, -0.0338] | 40/11/63 |
| F1-P1 | macro_f1 | 114 | -0.0883 | [-0.1234, -0.0567] | 37/5/72 |

## 平均秩

| 预算 | 指标 | 方法 | 平均秩 | 任务数 |
|---:|---|---|---:|---:|
| 1 | balanced_accuracy | R0 | 2.930 | 114 |
| 1 | balanced_accuracy | A0 | 2.610 | 114 |
| 1 | balanced_accuracy | F0 | 2.781 | 114 |
| 1 | balanced_accuracy | P0 | 1.680 | 114 |
| 1 | accuracy | R0 | 2.974 | 114 |
| 1 | accuracy | A0 | 2.469 | 114 |
| 1 | accuracy | F0 | 2.658 | 114 |
| 1 | accuracy | P0 | 1.899 | 114 |
| 1 | macro_f1 | R0 | 2.930 | 114 |
| 1 | macro_f1 | A0 | 2.579 | 114 |
| 1 | macro_f1 | F0 | 2.789 | 114 |
| 1 | macro_f1 | P0 | 1.702 | 114 |
| 3 | balanced_accuracy | R1 | 2.825 | 114 |
| 3 | balanced_accuracy | A1 | 2.702 | 114 |
| 3 | balanced_accuracy | F1 | 2.728 | 114 |
| 3 | balanced_accuracy | P1 | 1.746 | 114 |
| 3 | accuracy | R1 | 2.829 | 114 |
| 3 | accuracy | A1 | 2.544 | 114 |
| 3 | accuracy | F1 | 2.618 | 114 |
| 3 | accuracy | P1 | 2.009 | 114 |
| 3 | macro_f1 | R1 | 2.789 | 114 |
| 3 | macro_f1 | A1 | 2.684 | 114 |
| 3 | macro_f1 | F1 | 2.737 | 114 |
| 3 | macro_f1 | P1 | 1.789 | 114 |

## 配对共同完整任务

- A0-R0（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- A1-R1（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F0-R0（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F0-A0（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F0-P0（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F1-R1（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F1-A1（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga
- F1-P1（114 个）：ALLAML, Adiac, ArrowHead, CBF, CLL_SUB_111, Car, Carcinom, ChlorineConcentration, CinC_ECG_torso, Computers, Cricket_X, Cricket_Y, Cricket_Z, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, ECG200, ECG5000, Earthquakes, ElectricDevices, FISH, FaceAll, FaceFour, FacesUCR, FordA, FordB, GLIOMA, Gun_Point, Ham, HandOutlines, Haptics, Herring, InlineSkate, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lighting2, Lighting7, MALLAT, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFatalECG_Thorax1, NonInvasiveFatalECG_Thorax2, OSULeaf, PhalangesOutlinesCorrect, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, Two_Patterns, UWaveGestureLibraryAll, WordsSynonyms, Worms, WormsTwoClass, analcatdata_asbestos, analcatdata_bankruptcy, analcatdata_creditscore, analcatdata_fraud, analcatdata_japansolvent, backache, biomed, breast_cancer, bupa, cancer_cell_(cooh)2, cancer_cell_cooh, cancer_cell_nh2, chlorinated_samples, colon, comfile_stroke, corral, cspp_serum_metabolites, diabetes_skin_ear_lobe, diabetes_skin_inner_arm, diabetes_skin_thumbnail, diabetes_skin_vein, haberman, hair_dyes_sers, head_neck_cancer, hepatitis, irish, labor, locust_phase_hemolymph, lupus, marine_pathogens, marine_pathogens_binary, molecular_biology_promoters, mux6, pharmaceutical_ingredients, postoperative_patient_data, prnn_synth, saheart, serum_alzheimer_disease, synthetic_control, uWaveGestureLibrary_X, uWaveGestureLibrary_Y, uWaveGestureLibrary_Z, wafer, yoga

## 覆盖与失败

失败记录 0 项；无效选择 0 项。完整任务要求每个计划轮次的所有eligible模型均成功并共享相同Test样本。

## 费用字段

fit_duration_sum_seconds 是各 Run 自身 cost.wall_seconds 的总和，不是任务实际墙钟时间；每个 Run 的 wall_seconds 保存在 report.json 的 run_cost_details。
actual_cached_* 为实际复用记录；uncached_estimated_* 是基于缓存源拟合用时的估计，不代表实际重跑。LM token 只取 records.cost 明确给出的字段。

| 任务角色 | 预算 | 记录数 | 实际拟合数 | Fit duration sum (s) | 实际缓存拟合 | 实际缓存秒数 | 未缓存估计拟合 | 未缓存估计秒数 | 已知 LM token | 未知 token 数 | 缓存 token 估计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 1 | 305 | 515.000 | 1812.510 | 0.000 | 0.000 | 515.000 | 1812.510 | 61993.000 | 0.000 | 0.000 |
| development | 3 | 305 | 592.000 | 2731.730 | 470.000 | 4.111 | 1062.000 | 3754.389 | 61837.000 | 0.000 | 0.000 |
| locked | 1 | 2890 | 4570.000 | 126244.906 | 0.000 | 0.000 | 4570.000 | 126244.906 | 616214.000 | 0.000 | 0.000 |
| locked | 3 | 2890 | 6183.000 | 231622.681 | 3953.000 | 190.206 | 10136.000 | 322338.908 | 616392.000 | 0.000 | 0.000 |

### 任务实际墙钟时间

task_wall_seconds 来自 driver_state 的任务时钟；driver_state 未提供时保留为未知，不以 Fit duration sum 代替。

| 任务 | 角色 | task_wall_seconds | 云端核验时间 |
|---|---|---:|---|
| Beef | development | 6011.365 | 1791746936.3036504 |
| BeetleFly | development | 6011.593 | 1791746936.66673 |
| BirdChicken | development | 6011.826 | 1791746937.0329292 |
| Coffee | development | 6011.911 | 1791746937.295396 |
| ECGFiveDays | development | 6011.882 | 1791746937.6456614 |
| Meat | development | 6012.103 | 1791746938.0165505 |
| OliveOil | development | 6012.365 | 1791746938.3951943 |
| SonyAIBORobotSurface | development | 6012.508 | 1791746938.6558557 |
| SonyAIBORobotSurfaceII | development | 6012.757 | 1791746939.02436 |
| Wine | development | 6013.006 | 1791746939.4127412 |
| analcatdata_aids | development | 6013.064 | 1791746939.677314 |
| appendicitis | development | 6013.270 | 1791746940.04112 |
| parity5 | development | 6013.509 | 1791746940.4160914 |
| prnn_crabs | development | 6013.771 | 1791746940.8000164 |
| glass2 | development | 6013.882 | 1791746941.0643315 |
| ALLAML | locked | 31236.968 | — |
| CLL_SUB_111 | locked | 31236.791 | — |
| Carcinom | locked | 31236.617 | — |
| GLIOMA | locked | 31236.449 | — |
| colon | locked | 31236.273 | — |
| cancer_cell_(cooh)2 | locked | 31236.087 | — |
| cancer_cell_cooh | locked | 31235.905 | — |
| cancer_cell_nh2 | locked | 31235.705 | — |
| chlorinated_samples | locked | 31235.441 | — |
| comfile_stroke | locked | 31235.247 | — |
| cspp_serum_metabolites | locked | 31235.018 | — |
| diabetes_skin_ear_lobe | locked | 31234.817 | — |
| diabetes_skin_inner_arm | locked | 31234.574 | — |
| diabetes_skin_thumbnail | locked | 31234.367 | — |
| diabetes_skin_vein | locked | 31234.174 | — |
| hair_dyes_sers | locked | 31233.978 | — |
| head_neck_cancer | locked | 31233.777 | — |
| locust_phase_hemolymph | locked | 31233.574 | — |
| marine_pathogens | locked | 31233.372 | — |
| marine_pathogens_binary | locked | 31233.181 | — |
| pharmaceutical_ingredients | locked | 31232.986 | — |
| serum_alzheimer_disease | locked | 31232.801 | — |
| analcatdata_asbestos | locked | 31232.629 | — |
| analcatdata_bankruptcy | locked | 31232.448 | — |
| analcatdata_creditscore | locked | 31232.278 | — |
| analcatdata_fraud | locked | 31232.111 | — |
| analcatdata_japansolvent | locked | 31231.926 | — |
| backache | locked | 31231.759 | — |
| biomed | locked | 31231.582 | — |
| breast_cancer | locked | 31231.411 | — |
| bupa | locked | 31231.233 | — |
| corral | locked | 31231.052 | — |
| haberman | locked | 31230.873 | — |
| hepatitis | locked | 31230.698 | — |
| irish | locked | 31230.491 | — |
| labor | locked | 31230.325 | — |
| lupus | locked | 31230.157 | — |
| molecular_biology_promoters | locked | 31229.993 | — |
| mux6 | locked | 31229.818 | — |
| postoperative_patient_data | locked | 31229.651 | — |
| prnn_synth | locked | 31229.480 | — |
| saheart | locked | 31229.303 | — |
| Adiac | locked | 31229.143 | — |
| ArrowHead | locked | 31228.992 | — |
| CBF | locked | 31228.835 | — |
| Car | locked | 31228.679 | — |
| ChlorineConcentration | locked | 31228.519 | — |
| CinC_ECG_torso | locked | 31228.375 | — |
| Computers | locked | 31228.223 | — |
| Cricket_X | locked | 31228.065 | — |
| Cricket_Y | locked | 31227.902 | — |
| Cricket_Z | locked | 31227.752 | — |
| DistalPhalanxOutlineAgeGroup | locked | 31227.601 | — |
| DistalPhalanxOutlineCorrect | locked | 31227.441 | — |
| DistalPhalanxTW | locked | 31227.294 | — |
| ECG200 | locked | 31227.141 | — |
| ECG5000 | locked | 31226.988 | — |
| Earthquakes | locked | 31226.832 | — |
| ElectricDevices | locked | 31226.676 | — |
| FISH | locked | 31226.523 | — |
| FaceAll | locked | 31226.339 | — |
| FaceFour | locked | 31226.176 | — |
| FacesUCR | locked | 31226.024 | — |
| FordA | locked | 31225.873 | — |
| FordB | locked | 31225.716 | — |
| Gun_Point | locked | 31225.562 | — |
| Ham | locked | 31225.417 | — |
| HandOutlines | locked | 31225.267 | — |
| Haptics | locked | 31225.111 | — |
| Herring | locked | 31224.957 | — |
| InlineSkate | locked | 31224.798 | — |
| InsectWingbeatSound | locked | 31224.647 | — |
| ItalyPowerDemand | locked | 31224.489 | — |
| LargeKitchenAppliances | locked | 31224.332 | — |
| Lighting2 | locked | 31224.178 | — |
| Lighting7 | locked | 31223.960 | — |
| MALLAT | locked | 31223.808 | — |
| MedicalImages | locked | 31223.658 | — |
| MiddlePhalanxOutlineAgeGroup | locked | 31223.508 | — |
| MiddlePhalanxOutlineCorrect | locked | 31223.352 | — |
| MiddlePhalanxTW | locked | 31223.200 | — |
| MoteStrain | locked | 31223.045 | — |
| NonInvasiveFatalECG_Thorax1 | locked | 31222.891 | — |
| NonInvasiveFatalECG_Thorax2 | locked | 31222.730 | — |
| OSULeaf | locked | 31222.579 | — |
| PhalangesOutlinesCorrect | locked | 31222.419 | — |
| Plane | locked | 31222.248 | — |
| ProximalPhalanxOutlineAgeGroup | locked | 31222.092 | — |
| ProximalPhalanxOutlineCorrect | locked | 31221.944 | — |
| ProximalPhalanxTW | locked | 31221.787 | — |
| RefrigerationDevices | locked | 31221.637 | — |
| ScreenType | locked | 31221.486 | — |
| ShapeletSim | locked | 31221.322 | — |
| ShapesAll | locked | 31221.168 | — |
| SmallKitchenAppliances | locked | 31221.014 | — |
| StarLightCurves | locked | 31220.859 | — |
| Strawberry | locked | 31220.705 | — |
| SwedishLeaf | locked | 31220.555 | — |
| Symbols | locked | 31220.399 | — |
| ToeSegmentation1 | locked | 31220.249 | — |
| ToeSegmentation2 | locked | 31220.089 | — |
| Trace | locked | 31219.936 | — |
| TwoLeadECG | locked | 31219.774 | — |
| Two_Patterns | locked | 31219.623 | — |
| UWaveGestureLibraryAll | locked | 31219.463 | — |
| WordsSynonyms | locked | 31219.313 | — |
| Worms | locked | 31219.159 | — |
| WormsTwoClass | locked | 31219.003 | — |
| synthetic_control | locked | 31218.853 | — |
| uWaveGestureLibrary_X | locked | 31218.693 | — |
| uWaveGestureLibrary_Y | locked | 31218.538 | — |
| uWaveGestureLibrary_Z | locked | 31218.384 | — |
| wafer | locked | 31218.229 | — |
| yoga | locked | 31218.060 | — |
