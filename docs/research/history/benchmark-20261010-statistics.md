> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

# Benchmark 统计报告

正式统计范围为 114 个锁定分类任务；15 个开发任务仅用于确定固定模型，不进入正式效应统计。
正式 suite：biology5, raman34, tabmini25, ucr85。

R 每轮从当轮合法模型均匀抽取；balanced accuracy 与 accuracy 报告解析期望，多轮 Macro-F1 按固定 seed 模拟。P 仅依据 validation_score 选模型，Test 只用于计算最终分数。

## 方法结果

| 方法 | 预算 | 完整任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| R0 | 1 | 114/114 | 0.6689 | 0.6559 | 0.7141 |
| R1 | 3 | 114/114 | 0.6889 | 0.6780 | 0.7307 |
| A0 | 1 | 114/114 | 0.6635 | 0.6510 | 0.7063 |
| A1 | 3 | 114/114 | 0.6876 | 0.6779 | 0.7294 |
| F0 | 1 | 114/114 | 0.6718 | 0.6585 | 0.7200 |
| F1 | 3 | 114/114 | 0.6755 | 0.6619 | 0.7230 |
| P0 | 1 | 114/114 | 0.7526 | 0.7456 | 0.7866 |
| P1 | 3 | 114/114 | 0.7605 | 0.7532 | 0.7926 |

## 按 suite 汇总

| 方法 | 预算 | suite | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| R0 | 1 | biology5 | 5/5 | 0.7625 | 0.7749 | 0.7900 |
| R0 | 1 | raman34 | 17/17 | 0.6927 | 0.6858 | 0.7091 |
| R0 | 1 | tabmini25 | 20/20 | 0.7110 | 0.7085 | 0.7804 |
| R0 | 1 | ucr85 | 72/72 | 0.6451 | 0.6260 | 0.6916 |
| R1 | 3 | biology5 | 5/5 | 0.8099 | 0.8186 | 0.8280 |
| R1 | 3 | raman34 | 17/17 | 0.7230 | 0.7152 | 0.7370 |
| R1 | 3 | tabmini25 | 20/20 | 0.7145 | 0.7121 | 0.7830 |
| R1 | 3 | ucr85 | 72/72 | 0.6653 | 0.6500 | 0.7079 |
| A0 | 1 | biology5 | 5/5 | 0.8223 | 0.8328 | 0.8340 |
| A0 | 1 | raman34 | 17/17 | 0.7573 | 0.7517 | 0.7802 |
| A0 | 1 | tabmini25 | 20/20 | 0.7274 | 0.7264 | 0.7933 |
| A0 | 1 | ucr85 | 72/72 | 0.6126 | 0.5937 | 0.6559 |
| A1 | 3 | biology5 | 5/5 | 0.8304 | 0.8406 | 0.8421 |
| A1 | 3 | raman34 | 17/17 | 0.7730 | 0.7577 | 0.7904 |
| A1 | 3 | tabmini25 | 20/20 | 0.7174 | 0.7116 | 0.7890 |
| A1 | 3 | ucr85 | 72/72 | 0.6492 | 0.6384 | 0.6906 |
| F0 | 1 | biology5 | 5/5 | 0.8464 | 0.8509 | 0.8553 |
| F0 | 1 | raman34 | 17/17 | 0.7844 | 0.7726 | 0.8013 |
| F0 | 1 | tabmini25 | 20/20 | 0.6977 | 0.6903 | 0.7740 |
| F0 | 1 | ucr85 | 72/72 | 0.6259 | 0.6094 | 0.6764 |
| F1 | 3 | biology5 | 5/5 | 0.8464 | 0.8509 | 0.8553 |
| F1 | 3 | raman34 | 17/17 | 0.7835 | 0.7721 | 0.8003 |
| F1 | 3 | tabmini25 | 20/20 | 0.7073 | 0.6994 | 0.7814 |
| F1 | 3 | ucr85 | 72/72 | 0.6292 | 0.6123 | 0.6794 |
| P0 | 1 | biology5 | 5/5 | 0.8247 | 0.8309 | 0.8321 |
| P0 | 1 | raman34 | 17/17 | 0.7734 | 0.7616 | 0.7828 |
| P0 | 1 | tabmini25 | 20/20 | 0.7454 | 0.7431 | 0.8019 |
| P0 | 1 | ucr85 | 72/72 | 0.7446 | 0.7365 | 0.7802 |
| P1 | 3 | biology5 | 5/5 | 0.8386 | 0.8408 | 0.8415 |
| P1 | 3 | raman34 | 17/17 | 0.7837 | 0.7748 | 0.7899 |
| P1 | 3 | tabmini25 | 20/20 | 0.7516 | 0.7491 | 0.8075 |
| P1 | 3 | ucr85 | 72/72 | 0.7520 | 0.7431 | 0.7857 |

## 按 family 汇总

| 方法 | 预算 | family_id | 完整/预期任务 | Balanced accuracy | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|---:|---:|
| R0 | 1 | biology5:ALLAML | 1/1 | 0.9063 | 0.9158 | 0.9266 |
| R0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.7648 | 0.7712 | 0.7104 |
| R0 | 1 | biology5:Carcinom | 1/1 | 0.6597 | 0.6873 | 0.7537 |
| R0 | 1 | biology5:GLIOMA | 1/1 | 0.7218 | 0.7314 | 0.7600 |
| R0 | 1 | biology5:colon | 1/1 | 0.7599 | 0.7687 | 0.7995 |
| R0 | 1 | raman34:cancer_cell_media | 3/3 | 0.7140 | 0.7164 | 0.7190 |
| R0 | 1 | raman34:chlorinated_samples | 1/1 | 0.8680 | 0.8761 | 0.8963 |
| R0 | 1 | raman34:comfile_stroke | 1/1 | 0.7962 | 0.7943 | 0.7962 |
| R0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 0.8923 | 0.8890 | 0.8923 |
| R0 | 1 | raman34:diabetes_skin | 4/4 | 0.4870 | 0.4588 | 0.5077 |
| R0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.8687 | 0.8922 | 0.9070 |
| R0 | 1 | raman34:head_neck_cancer | 1/1 | 0.3953 | 0.3716 | 0.4664 |
| R0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.8392 | 0.8387 | 0.8392 |
| R0 | 1 | raman34:marine_pathogens | 2/2 | 0.9485 | 0.9499 | 0.9510 |
| R0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.8673 | 0.8838 | 0.8689 |
| R0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.2623 | 0.2280 | 0.2986 |
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
| R0 | 1 | ucr85:Adiac | 1/1 | 0.6084 | 0.5672 | 0.5924 |
| R0 | 1 | ucr85:ArrowHead | 1/1 | 0.6497 | 0.6413 | 0.6453 |
| R0 | 1 | ucr85:CBF | 1/1 | 0.8226 | 0.8171 | 0.8220 |
| R0 | 1 | ucr85:Car | 1/1 | 0.6109 | 0.5969 | 0.6115 |
| R0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.6347 | 0.6288 | 0.7175 |
| R0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.5608 | 0.5514 | 0.5599 |
| R0 | 1 | ucr85:Computers | 1/1 | 0.6194 | 0.6165 | 0.6194 |
| R0 | 1 | ucr85:Cricket | 3/3 | 0.5211 | 0.4991 | 0.5171 |
| R0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5752 | 0.5574 | 0.7927 |
| R0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7081 | 0.7072 | 0.7309 |
| R0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4360 | 0.4010 | 0.7540 |
| R0 | 1 | ucr85:ECG200 | 1/1 | 0.8048 | 0.8083 | 0.8262 |
| R0 | 1 | ucr85:ECG5000 | 1/1 | 0.4318 | 0.4297 | 0.9136 |
| R0 | 1 | ucr85:Earthquakes | 1/1 | 0.5169 | 0.4789 | 0.7831 |
| R0 | 1 | ucr85:ElectricDevices | 1/1 | 0.4963 | 0.4806 | 0.6082 |
| R0 | 1 | ucr85:FISH | 1/1 | 0.7319 | 0.7135 | 0.7301 |
| R0 | 1 | ucr85:FaceAll | 1/1 | 0.7525 | 0.6866 | 0.7048 |
| R0 | 1 | ucr85:FaceFour | 1/1 | 0.6302 | 0.5604 | 0.6128 |
| R0 | 1 | ucr85:FacesUCR | 1/1 | 0.5928 | 0.5892 | 0.6618 |
| R0 | 1 | ucr85:FordA | 1/1 | 0.7293 | 0.7083 | 0.7316 |
| R0 | 1 | ucr85:FordB | 1/1 | 0.7022 | 0.7009 | 0.7019 |
| R0 | 1 | ucr85:Gun_Point | 1/1 | 0.8614 | 0.8610 | 0.8615 |
| R0 | 1 | ucr85:Ham | 1/1 | 0.6324 | 0.6177 | 0.6359 |
| R0 | 1 | ucr85:HandOutlines | 1/1 | 0.8083 | 0.8133 | 0.8302 |
| R0 | 1 | ucr85:Haptics | 1/1 | 0.4024 | 0.3616 | 0.4073 |
| R0 | 1 | ucr85:Herring | 1/1 | 0.5326 | 0.4548 | 0.5781 |
| R0 | 1 | ucr85:InlineSkate | 1/1 | 0.3052 | 0.2906 | 0.3078 |
| R0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.5350 | 0.5190 | 0.5350 |
| R0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9435 | 0.9435 | 0.9435 |
| R0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.6228 | 0.6216 | 0.6228 |
| R0 | 1 | ucr85:Lighting2 | 1/1 | 0.6928 | 0.6885 | 0.7062 |
| R0 | 1 | ucr85:Lighting7 | 1/1 | 0.5333 | 0.4733 | 0.5627 |
| R0 | 1 | ucr85:MALLAT | 1/1 | 0.7334 | 0.6910 | 0.7315 |
| R0 | 1 | ucr85:MedicalImages | 1/1 | 0.3823 | 0.3815 | 0.6079 |
| R0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.5745 | 0.5165 | 0.6706 |
| R0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5939 | 0.5500 | 0.5527 |
| R0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3636 | 0.3218 | 0.6156 |
| R0 | 1 | ucr85:MoteStrain | 1/1 | 0.7340 | 0.6973 | 0.7407 |
| R0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.7805 | 0.7641 | 0.7800 |
| R0 | 1 | ucr85:OSULeaf | 1/1 | 0.6317 | 0.6138 | 0.6446 |
| R0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.6888 | 0.6722 | 0.7287 |
| R0 | 1 | ucr85:Plane | 1/1 | 0.9465 | 0.9384 | 0.9436 |
| R0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7679 | 0.7398 | 0.8304 |
| R0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7911 | 0.8031 | 0.8440 |
| R0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.4468 | 0.4216 | 0.7873 |
| R0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.4607 | 0.4534 | 0.4607 |
| R0 | 1 | ucr85:ScreenType | 1/1 | 0.4459 | 0.4391 | 0.4459 |
| R0 | 1 | ucr85:ShapeletSim | 1/1 | 0.6269 | 0.6028 | 0.6269 |
| R0 | 1 | ucr85:ShapesAll | 1/1 | 0.6497 | 0.6205 | 0.6497 |
| R0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.6111 | 0.6020 | 0.6111 |
| R0 | 1 | ucr85:StarLightCurves | 1/1 | 0.8818 | 0.8766 | 0.9270 |
| R0 | 1 | ucr85:Strawberry | 1/1 | 0.9135 | 0.9087 | 0.9235 |
| R0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.8134 | 0.7991 | 0.8086 |
| R0 | 1 | ucr85:Symbols | 1/1 | 0.7255 | 0.7059 | 0.7273 |
| R0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.7317 | 0.7295 | 0.7298 |
| R0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.7023 | 0.6462 | 0.7065 |
| R0 | 1 | ucr85:Trace | 1/1 | 0.8774 | 0.8451 | 0.8646 |
| R0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.7364 | 0.7190 | 0.7363 |
| R0 | 1 | ucr85:Two_Patterns | 1/1 | 0.8136 | 0.8087 | 0.8137 |
| R0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.2136 | 0.1872 | 0.4400 |
| R0 | 1 | ucr85:Worms | 1/1 | 0.3323 | 0.3086 | 0.4407 |
| R0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5830 | 0.5668 | 0.6239 |
| R0 | 1 | ucr85:synthetic_control | 1/1 | 0.9123 | 0.9069 | 0.9123 |
| R0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.7009 | 0.6814 | 0.7002 |
| R0 | 1 | ucr85:wafer | 1/1 | 0.6495 | 0.6673 | 0.9222 |
| R0 | 1 | ucr85:yoga | 1/1 | 0.7005 | 0.6986 | 0.7062 |
| R1 | 3 | biology5:ALLAML | 1/1 | 0.9332 | 0.9346 | 0.9425 |
| R1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.7939 | 0.7988 | 0.7362 |
| R1 | 3 | biology5:Carcinom | 1/1 | 0.7853 | 0.8057 | 0.8571 |
| R1 | 3 | biology5:GLIOMA | 1/1 | 0.7367 | 0.7468 | 0.7771 |
| R1 | 3 | biology5:colon | 1/1 | 0.8003 | 0.8069 | 0.8272 |
| R1 | 3 | raman34:cancer_cell_media | 3/3 | 0.7776 | 0.7793 | 0.7817 |
| R1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9061 | 0.9124 | 0.9237 |
| R1 | 3 | raman34:comfile_stroke | 1/1 | 0.8192 | 0.8184 | 0.8192 |
| R1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 0.9282 | 0.9274 | 0.9282 |
| R1 | 3 | raman34:diabetes_skin | 4/4 | 0.4861 | 0.4598 | 0.5067 |
| R1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9493 | 0.9591 | 0.9628 |
| R1 | 3 | raman34:head_neck_cancer | 1/1 | 0.4282 | 0.4139 | 0.4997 |
| R1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.8781 | 0.8781 | 0.8781 |
| R1 | 3 | raman34:marine_pathogens | 2/2 | 0.9678 | 0.9678 | 0.9685 |
| R1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9186 | 0.9296 | 0.9192 |
| R1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2505 | 0.2062 | 0.2896 |
| R1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8103 | 0.8078 | 0.8090 |
| R1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8400 | 0.8401 | 0.8400 |
| R1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.7654 | 0.7805 | 0.8400 |
| R1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5623 | 0.5526 | 0.6803 |
| R1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8020 | 0.8017 | 0.8049 |
| R1 | 3 | tabmini25:backache | 1/1 | 0.5098 | 0.4989 | 0.8325 |
| R1 | 3 | tabmini25:biomed | 1/1 | 0.8348 | 0.8482 | 0.8667 |
| R1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5925 | 0.5937 | 0.7238 |
| R1 | 3 | tabmini25:bupa | 1/1 | 0.6128 | 0.6106 | 0.6116 |
| R1 | 3 | tabmini25:corral | 1/1 | 0.9037 | 0.9050 | 0.9062 |
| R1 | 3 | tabmini25:haberman | 1/1 | 0.5351 | 0.5136 | 0.7297 |
| R1 | 3 | tabmini25:hepatitis | 1/1 | 0.6607 | 0.6803 | 0.8258 |
| R1 | 3 | tabmini25:irish | 1/1 | 0.8500 | 0.8505 | 0.8534 |
| R1 | 3 | tabmini25:labor | 1/1 | 0.8400 | 0.8473 | 0.8647 |
| R1 | 3 | tabmini25:lupus | 1/1 | 0.7326 | 0.7369 | 0.7537 |
| R1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.8046 | 0.8051 | 0.8046 |
| R1 | 3 | tabmini25:mux6 | 1/1 | 0.6429 | 0.6416 | 0.6429 |
| R1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.4970 | 0.4279 | 0.7175 |
| R1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8480 | 0.8480 | 0.8480 |
| R1 | 3 | tabmini25:saheart | 1/1 | 0.6463 | 0.6519 | 0.7044 |
| R1 | 3 | ucr85:Adiac | 1/1 | 0.6670 | 0.6282 | 0.6541 |
| R1 | 3 | ucr85:ArrowHead | 1/1 | 0.6455 | 0.6381 | 0.6418 |
| R1 | 3 | ucr85:CBF | 1/1 | 0.8379 | 0.8362 | 0.8375 |
| R1 | 3 | ucr85:Car | 1/1 | 0.6290 | 0.6105 | 0.6282 |
| R1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.6795 | 0.6855 | 0.7465 |
| R1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.5742 | 0.5647 | 0.5733 |
| R1 | 3 | ucr85:Computers | 1/1 | 0.6215 | 0.6202 | 0.6215 |
| R1 | 3 | ucr85:Cricket | 3/3 | 0.5496 | 0.5379 | 0.5465 |
| R1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5865 | 0.5733 | 0.7873 |
| R1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7343 | 0.7334 | 0.7517 |
| R1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4519 | 0.4257 | 0.7604 |
| R1 | 3 | ucr85:ECG200 | 1/1 | 0.7891 | 0.7940 | 0.8138 |
| R1 | 3 | ucr85:ECG5000 | 1/1 | 0.4451 | 0.4405 | 0.9252 |
| R1 | 3 | ucr85:Earthquakes | 1/1 | 0.5483 | 0.5008 | 0.7200 |
| R1 | 3 | ucr85:ElectricDevices | 1/1 | 0.5105 | 0.5018 | 0.6161 |
| R1 | 3 | ucr85:FISH | 1/1 | 0.7644 | 0.7561 | 0.7631 |
| R1 | 3 | ucr85:FaceAll | 1/1 | 0.8081 | 0.7480 | 0.7626 |
| R1 | 3 | ucr85:FaceFour | 1/1 | 0.6330 | 0.5616 | 0.6145 |
| R1 | 3 | ucr85:FacesUCR | 1/1 | 0.6460 | 0.6462 | 0.7122 |
| R1 | 3 | ucr85:FordA | 1/1 | 0.7476 | 0.7341 | 0.7491 |
| R1 | 3 | ucr85:FordB | 1/1 | 0.7064 | 0.7053 | 0.7060 |
| R1 | 3 | ucr85:Gun_Point | 1/1 | 0.8745 | 0.8745 | 0.8749 |
| R1 | 3 | ucr85:Ham | 1/1 | 0.6300 | 0.6170 | 0.6330 |
| R1 | 3 | ucr85:HandOutlines | 1/1 | 0.8229 | 0.8314 | 0.8496 |
| R1 | 3 | ucr85:Haptics | 1/1 | 0.4139 | 0.3813 | 0.4168 |
| R1 | 3 | ucr85:Herring | 1/1 | 0.5341 | 0.4564 | 0.5793 |
| R1 | 3 | ucr85:InlineSkate | 1/1 | 0.3100 | 0.3017 | 0.3113 |
| R1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5596 | 0.5474 | 0.5596 |
| R1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9435 | 0.9434 | 0.9435 |
| R1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.6332 | 0.6322 | 0.6332 |
| R1 | 3 | ucr85:Lighting2 | 1/1 | 0.6852 | 0.6813 | 0.6986 |
| R1 | 3 | ucr85:Lighting7 | 1/1 | 0.5793 | 0.5339 | 0.5890 |
| R1 | 3 | ucr85:MALLAT | 1/1 | 0.7357 | 0.6940 | 0.7338 |
| R1 | 3 | ucr85:MedicalImages | 1/1 | 0.4424 | 0.4535 | 0.6333 |
| R1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.5658 | 0.5076 | 0.6673 |
| R1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.6052 | 0.5646 | 0.5677 |
| R1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3718 | 0.3343 | 0.6133 |
| R1 | 3 | ucr85:MoteStrain | 1/1 | 0.7280 | 0.6973 | 0.7322 |
| R1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.8561 | 0.8457 | 0.8572 |
| R1 | 3 | ucr85:OSULeaf | 1/1 | 0.6506 | 0.6351 | 0.6621 |
| R1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7181 | 0.7118 | 0.7488 |
| R1 | 3 | ucr85:Plane | 1/1 | 0.9806 | 0.9794 | 0.9802 |
| R1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7668 | 0.7453 | 0.8368 |
| R1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8123 | 0.8214 | 0.8530 |
| R1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4581 | 0.4381 | 0.7917 |
| R1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.4615 | 0.4544 | 0.4615 |
| R1 | 3 | ucr85:ScreenType | 1/1 | 0.4474 | 0.4424 | 0.4474 |
| R1 | 3 | ucr85:ShapeletSim | 1/1 | 0.6278 | 0.6036 | 0.6278 |
| R1 | 3 | ucr85:ShapesAll | 1/1 | 0.6931 | 0.6693 | 0.6931 |
| R1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.6176 | 0.6102 | 0.6176 |
| R1 | 3 | ucr85:StarLightCurves | 1/1 | 0.8955 | 0.8902 | 0.9325 |
| R1 | 3 | ucr85:Strawberry | 1/1 | 0.9498 | 0.9463 | 0.9503 |
| R1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.8585 | 0.8494 | 0.8545 |
| R1 | 3 | ucr85:Symbols | 1/1 | 0.7284 | 0.7088 | 0.7303 |
| R1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.7360 | 0.7339 | 0.7341 |
| R1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.6953 | 0.6411 | 0.7012 |
| R1 | 3 | ucr85:Trace | 1/1 | 0.8833 | 0.8614 | 0.8731 |
| R1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.7377 | 0.7207 | 0.7376 |
| R1 | 3 | ucr85:Two_Patterns | 1/1 | 0.8994 | 0.8989 | 0.8992 |
| R1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.2553 | 0.2364 | 0.4691 |
| R1 | 3 | ucr85:Worms | 1/1 | 0.3455 | 0.3223 | 0.4373 |
| R1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.5992 | 0.5889 | 0.6362 |
| R1 | 3 | ucr85:synthetic_control | 1/1 | 0.9479 | 0.9453 | 0.9479 |
| R1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7329 | 0.7204 | 0.7324 |
| R1 | 3 | ucr85:wafer | 1/1 | 0.6684 | 0.6929 | 0.9251 |
| R1 | 3 | ucr85:yoga | 1/1 | 0.7113 | 0.7099 | 0.7162 |
| A0 | 1 | biology5:ALLAML | 1/1 | 0.9494 | 0.9536 | 0.9583 |
| A0 | 1 | biology5:CLL_SUB_111 | 1/1 | 0.8062 | 0.8062 | 0.7387 |
| A0 | 1 | biology5:Carcinom | 1/1 | 0.7858 | 0.8135 | 0.8506 |
| A0 | 1 | biology5:GLIOMA | 1/1 | 0.7690 | 0.7867 | 0.8000 |
| A0 | 1 | biology5:colon | 1/1 | 0.8011 | 0.8042 | 0.8226 |
| A0 | 1 | raman34:cancer_cell_media | 3/3 | 0.8744 | 0.8783 | 0.8765 |
| A0 | 1 | raman34:chlorinated_samples | 1/1 | 0.9179 | 0.9292 | 0.9391 |
| A0 | 1 | raman34:comfile_stroke | 1/1 | 0.8500 | 0.8496 | 0.8500 |
| A0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| A0 | 1 | raman34:diabetes_skin | 4/4 | 0.5619 | 0.5332 | 0.5875 |
| A0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.6948 | 0.7304 | 0.8044 |
| A0 | 1 | raman34:head_neck_cancer | 1/1 | 0.5045 | 0.5022 | 0.5946 |
| A0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.8450 | 0.8449 | 0.8450 |
| A0 | 1 | raman34:marine_pathogens | 2/2 | 0.9532 | 0.9561 | 0.9585 |
| A0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.9806 | 0.9814 | 0.9806 |
| A0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.3048 | 0.2619 | 0.3529 |
| A0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.8211 | 0.8183 | 0.8193 |
| A0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8400 | 0.8400 | 0.8400 |
| A0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.8059 | 0.8243 | 0.8700 |
| A0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.5464 | 0.5406 | 0.6667 |
| A0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8444 | 0.8452 | 0.8462 |
| A0 | 1 | tabmini25:backache | 1/1 | 0.5013 | 0.4968 | 0.8056 |
| A0 | 1 | tabmini25:biomed | 1/1 | 0.8398 | 0.8496 | 0.8660 |
| A0 | 1 | tabmini25:breast_cancer | 1/1 | 0.6088 | 0.6145 | 0.7413 |
| A0 | 1 | tabmini25:bupa | 1/1 | 0.6153 | 0.6142 | 0.6145 |
| A0 | 1 | tabmini25:corral | 1/1 | 0.8897 | 0.8915 | 0.8938 |
| A0 | 1 | tabmini25:haberman | 1/1 | 0.5360 | 0.5078 | 0.7418 |
| A0 | 1 | tabmini25:hepatitis | 1/1 | 0.7453 | 0.7758 | 0.8710 |
| A0 | 1 | tabmini25:irish | 1/1 | 0.8911 | 0.8923 | 0.8940 |
| A0 | 1 | tabmini25:labor | 1/1 | 0.8615 | 0.8782 | 0.8947 |
| A0 | 1 | tabmini25:lupus | 1/1 | 0.7374 | 0.7421 | 0.7586 |
| A0 | 1 | tabmini25:molecular_biology_promoters | 1/1 | 0.8113 | 0.8113 | 0.8113 |
| A0 | 1 | tabmini25:mux6 | 1/1 | 0.6250 | 0.6250 | 0.6250 |
| A0 | 1 | tabmini25:postoperative_patient_data | 1/1 | 0.4922 | 0.4172 | 0.7159 |
| A0 | 1 | tabmini25:prnn_synth | 1/1 | 0.8560 | 0.8560 | 0.8560 |
| A0 | 1 | tabmini25:saheart | 1/1 | 0.6788 | 0.6867 | 0.7338 |
| A0 | 1 | ucr85:Adiac | 1/1 | 0.7289 | 0.7007 | 0.7263 |
| A0 | 1 | ucr85:ArrowHead | 1/1 | 0.6591 | 0.6562 | 0.6571 |
| A0 | 1 | ucr85:CBF | 1/1 | 0.8837 | 0.8838 | 0.8833 |
| A0 | 1 | ucr85:Car | 1/1 | 0.6736 | 0.6768 | 0.6667 |
| A0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.7324 | 0.7541 | 0.7812 |
| A0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.6881 | 0.6723 | 0.6862 |
| A0 | 1 | ucr85:Computers | 1/1 | 0.6000 | 0.5999 | 0.6000 |
| A0 | 1 | ucr85:Cricket | 3/3 | 0.4298 | 0.3800 | 0.4145 |
| A0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5523 | 0.5488 | 0.7750 |
| A0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7800 | 0.7877 | 0.8083 |
| A0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4769 | 0.4268 | 0.7950 |
| A0 | 1 | ucr85:ECG200 | 1/1 | 0.8385 | 0.8302 | 0.8400 |
| A0 | 1 | ucr85:ECG5000 | 1/1 | 0.4340 | 0.4498 | 0.9289 |
| A0 | 1 | ucr85:Earthquakes | 1/1 | 0.4937 | 0.4749 | 0.7764 |
| A0 | 1 | ucr85:ElectricDevices | 1/1 | 0.3680 | 0.3424 | 0.4911 |
| A0 | 1 | ucr85:FISH | 1/1 | 0.8541 | 0.8503 | 0.8514 |
| A0 | 1 | ucr85:FaceAll | 1/1 | 0.8108 | 0.7513 | 0.7669 |
| A0 | 1 | ucr85:FaceFour | 1/1 | 0.8711 | 0.8651 | 0.8636 |
| A0 | 1 | ucr85:FacesUCR | 1/1 | 0.5957 | 0.6082 | 0.6839 |
| A0 | 1 | ucr85:FordA | 1/1 | 0.5000 | 0.3389 | 0.5126 |
| A0 | 1 | ucr85:FordB | 1/1 | 0.5415 | 0.5383 | 0.5399 |
| A0 | 1 | ucr85:Gun_Point | 1/1 | 0.9059 | 0.9062 | 0.9067 |
| A0 | 1 | ucr85:Ham | 1/1 | 0.7326 | 0.7327 | 0.7333 |
| A0 | 1 | ucr85:HandOutlines | 1/1 | 0.8194 | 0.8341 | 0.8550 |
| A0 | 1 | ucr85:Haptics | 1/1 | 0.4624 | 0.4268 | 0.4643 |
| A0 | 1 | ucr85:Herring | 1/1 | 0.5172 | 0.4868 | 0.5781 |
| A0 | 1 | ucr85:InlineSkate | 1/1 | 0.1859 | 0.1462 | 0.1945 |
| A0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.6364 | 0.6372 | 0.6364 |
| A0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9533 | 0.9533 | 0.9534 |
| A0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.5920 | 0.5912 | 0.5920 |
| A0 | 1 | ucr85:Lighting2 | 1/1 | 0.6055 | 0.5951 | 0.6230 |
| A0 | 1 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| A0 | 1 | ucr85:MALLAT | 1/1 | 0.8268 | 0.7834 | 0.8247 |
| A0 | 1 | ucr85:MedicalImages | 1/1 | 0.3825 | 0.4273 | 0.6118 |
| A0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.4174 | 0.2978 | 0.4225 |
| A0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5876 | 0.5400 | 0.5400 |
| A0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3890 | 0.3584 | 0.6366 |
| A0 | 1 | ucr85:MoteStrain | 1/1 | 0.8603 | 0.8608 | 0.8618 |
| A0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.5548 | 0.4843 | 0.5394 |
| A0 | 1 | ucr85:OSULeaf | 1/1 | 0.3640 | 0.2946 | 0.3719 |
| A0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7192 | 0.7292 | 0.7681 |
| A0 | 1 | ucr85:Plane | 1/1 | 0.9726 | 0.9724 | 0.9714 |
| A0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8017 | 0.7890 | 0.8634 |
| A0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8101 | 0.8220 | 0.8522 |
| A0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.5161 | 0.5114 | 0.7925 |
| A0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.3707 | 0.3709 | 0.3707 |
| A0 | 1 | ucr85:ScreenType | 1/1 | 0.4480 | 0.4244 | 0.4480 |
| A0 | 1 | ucr85:ShapeletSim | 1/1 | 0.4833 | 0.4797 | 0.4833 |
| A0 | 1 | ucr85:ShapesAll | 1/1 | 0.3517 | 0.2944 | 0.3517 |
| A0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.4640 | 0.4586 | 0.4640 |
| A0 | 1 | ucr85:StarLightCurves | 1/1 | 0.8842 | 0.8886 | 0.9246 |
| A0 | 1 | ucr85:Strawberry | 1/1 | 0.9203 | 0.9186 | 0.9250 |
| A0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.6217 | 0.6149 | 0.6160 |
| A0 | 1 | ucr85:Symbols | 1/1 | 0.3699 | 0.2884 | 0.3678 |
| A0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.5819 | 0.5787 | 0.5789 |
| A0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.3797 | 0.3765 | 0.4615 |
| A0 | 1 | ucr85:Trace | 1/1 | 0.7768 | 0.6992 | 0.7500 |
| A0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.8253 | 0.8248 | 0.8253 |
| A0 | 1 | ucr85:Two_Patterns | 1/1 | 0.3207 | 0.2846 | 0.3242 |
| A0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.2588 | 0.2567 | 0.4796 |
| A0 | 1 | ucr85:Worms | 1/1 | 0.3078 | 0.3020 | 0.4088 |
| A0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5685 | 0.5519 | 0.6133 |
| A0 | 1 | ucr85:synthetic_control | 1/1 | 0.8733 | 0.8724 | 0.8733 |
| A0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.6655 | 0.6499 | 0.6651 |
| A0 | 1 | ucr85:wafer | 1/1 | 0.7681 | 0.8132 | 0.9388 |
| A0 | 1 | ucr85:yoga | 1/1 | 0.7054 | 0.7024 | 0.7183 |
| A1 | 3 | biology5:ALLAML | 1/1 | 0.9800 | 0.9845 | 0.9861 |
| A1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.8062 | 0.8062 | 0.7387 |
| A1 | 3 | biology5:Carcinom | 1/1 | 0.8186 | 0.8447 | 0.8793 |
| A1 | 3 | biology5:GLIOMA | 1/1 | 0.7690 | 0.7836 | 0.8000 |
| A1 | 3 | biology5:colon | 1/1 | 0.7784 | 0.7840 | 0.8065 |
| A1 | 3 | raman34:cancer_cell_media | 3/3 | 0.9257 | 0.9263 | 0.9259 |
| A1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9016 | 0.9097 | 0.9217 |
| A1 | 3 | raman34:comfile_stroke | 1/1 | 0.8500 | 0.8496 | 0.8500 |
| A1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| A1 | 3 | raman34:diabetes_skin | 4/4 | 0.5290 | 0.4678 | 0.5625 |
| A1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9613 | 0.9690 | 0.9714 |
| A1 | 3 | raman34:head_neck_cancer | 1/1 | 0.5208 | 0.5095 | 0.6216 |
| A1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.8750 | 0.8750 | 0.8750 |
| A1 | 3 | raman34:marine_pathogens | 2/2 | 0.9757 | 0.9774 | 0.9781 |
| A1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9779 | 0.9783 | 0.9783 |
| A1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2095 | 0.1843 | 0.2353 |
| A1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8428 | 0.8419 | 0.8434 |
| A1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8200 | 0.8199 | 0.8200 |
| A1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.8128 | 0.8355 | 0.8800 |
| A1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5212 | 0.4725 | 0.6905 |
| A1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8259 | 0.8263 | 0.8269 |
| A1 | 3 | tabmini25:backache | 1/1 | 0.4839 | 0.4545 | 0.8333 |
| A1 | 3 | tabmini25:biomed | 1/1 | 0.8472 | 0.8594 | 0.8756 |
| A1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5568 | 0.5458 | 0.7063 |
| A1 | 3 | tabmini25:bupa | 1/1 | 0.6274 | 0.6251 | 0.6261 |
| A1 | 3 | tabmini25:corral | 1/1 | 0.8786 | 0.8792 | 0.8812 |
| A1 | 3 | tabmini25:haberman | 1/1 | 0.5131 | 0.4721 | 0.7255 |
| A1 | 3 | tabmini25:hepatitis | 1/1 | 0.6822 | 0.7016 | 0.8258 |
| A1 | 3 | tabmini25:irish | 1/1 | 0.9136 | 0.9147 | 0.9160 |
| A1 | 3 | tabmini25:labor | 1/1 | 0.7959 | 0.8026 | 0.8246 |
| A1 | 3 | tabmini25:lupus | 1/1 | 0.7563 | 0.7586 | 0.7701 |
| A1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.8208 | 0.8204 | 0.8208 |
| A1 | 3 | tabmini25:mux6 | 1/1 | 0.6406 | 0.6405 | 0.6406 |
| A1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.5052 | 0.4531 | 0.7159 |
| A1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8480 | 0.8480 | 0.8480 |
| A1 | 3 | tabmini25:saheart | 1/1 | 0.6547 | 0.6607 | 0.7100 |
| A1 | 3 | ucr85:Adiac | 1/1 | 0.6196 | 0.5589 | 0.6036 |
| A1 | 3 | ucr85:ArrowHead | 1/1 | 0.7467 | 0.7314 | 0.7314 |
| A1 | 3 | ucr85:CBF | 1/1 | 0.8917 | 0.8909 | 0.8911 |
| A1 | 3 | ucr85:Car | 1/1 | 0.8347 | 0.8363 | 0.8333 |
| A1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.5167 | 0.5149 | 0.6547 |
| A1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.6164 | 0.5952 | 0.6145 |
| A1 | 3 | ucr85:Computers | 1/1 | 0.5080 | 0.5051 | 0.5080 |
| A1 | 3 | ucr85:Cricket | 3/3 | 0.5113 | 0.5055 | 0.5060 |
| A1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.7163 | 0.7568 | 0.8450 |
| A1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.8100 | 0.8035 | 0.8133 |
| A1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.5105 | 0.4851 | 0.8050 |
| A1 | 3 | ucr85:ECG200 | 1/1 | 0.7726 | 0.7835 | 0.8100 |
| A1 | 3 | ucr85:ECG5000 | 1/1 | 0.5447 | 0.5463 | 0.9204 |
| A1 | 3 | ucr85:Earthquakes | 1/1 | 0.5138 | 0.5071 | 0.7764 |
| A1 | 3 | ucr85:ElectricDevices | 1/1 | 0.5624 | 0.5714 | 0.6650 |
| A1 | 3 | ucr85:FISH | 1/1 | 0.7606 | 0.7554 | 0.7543 |
| A1 | 3 | ucr85:FaceAll | 1/1 | 0.7973 | 0.7340 | 0.7118 |
| A1 | 3 | ucr85:FaceFour | 1/1 | 0.8711 | 0.8651 | 0.8636 |
| A1 | 3 | ucr85:FacesUCR | 1/1 | 0.6730 | 0.6848 | 0.7663 |
| A1 | 3 | ucr85:FordA | 1/1 | 0.6776 | 0.6776 | 0.6779 |
| A1 | 3 | ucr85:FordB | 1/1 | 0.5505 | 0.5484 | 0.5492 |
| A1 | 3 | ucr85:Gun_Point | 1/1 | 0.8588 | 0.8586 | 0.8600 |
| A1 | 3 | ucr85:Ham | 1/1 | 0.5692 | 0.5675 | 0.5714 |
| A1 | 3 | ucr85:HandOutlines | 1/1 | 0.8019 | 0.8169 | 0.8410 |
| A1 | 3 | ucr85:Haptics | 1/1 | 0.4624 | 0.4268 | 0.4643 |
| A1 | 3 | ucr85:Herring | 1/1 | 0.5678 | 0.5622 | 0.6094 |
| A1 | 3 | ucr85:InlineSkate | 1/1 | 0.3254 | 0.3256 | 0.3145 |
| A1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5965 | 0.5864 | 0.5965 |
| A1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9486 | 0.9484 | 0.9485 |
| A1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.4000 | 0.3999 | 0.4000 |
| A1 | 3 | ucr85:Lighting2 | 1/1 | 0.6634 | 0.6420 | 0.6885 |
| A1 | 3 | ucr85:Lighting7 | 1/1 | 0.6727 | 0.6048 | 0.6712 |
| A1 | 3 | ucr85:MALLAT | 1/1 | 0.7529 | 0.7157 | 0.7510 |
| A1 | 3 | ucr85:MedicalImages | 1/1 | 0.5163 | 0.5852 | 0.6908 |
| A1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.5750 | 0.5761 | 0.7175 |
| A1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5825 | 0.5333 | 0.5333 |
| A1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3309 | 0.2781 | 0.5965 |
| A1 | 3 | ucr85:MoteStrain | 1/1 | 0.8685 | 0.8683 | 0.8690 |
| A1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9154 | 0.9136 | 0.9165 |
| A1 | 3 | ucr85:OSULeaf | 1/1 | 0.4397 | 0.3980 | 0.4504 |
| A1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.5982 | 0.5926 | 0.6585 |
| A1 | 3 | ucr85:Plane | 1/1 | 0.9815 | 0.9815 | 0.9810 |
| A1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7235 | 0.7151 | 0.8244 |
| A1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7738 | 0.7950 | 0.8385 |
| A1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4447 | 0.4063 | 0.8100 |
| A1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.3653 | 0.3652 | 0.3653 |
| A1 | 3 | ucr85:ScreenType | 1/1 | 0.3787 | 0.3768 | 0.3787 |
| A1 | 3 | ucr85:ShapeletSim | 1/1 | 0.4778 | 0.4725 | 0.4778 |
| A1 | 3 | ucr85:ShapesAll | 1/1 | 0.6683 | 0.6373 | 0.6683 |
| A1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.4640 | 0.4586 | 0.4640 |
| A1 | 3 | ucr85:StarLightCurves | 1/1 | 0.9110 | 0.9191 | 0.9446 |
| A1 | 3 | ucr85:Strawberry | 1/1 | 0.9203 | 0.9186 | 0.9250 |
| A1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.8417 | 0.8347 | 0.8352 |
| A1 | 3 | ucr85:Symbols | 1/1 | 0.6849 | 0.6799 | 0.6884 |
| A1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.5588 | 0.5570 | 0.5570 |
| A1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.3797 | 0.3765 | 0.4615 |
| A1 | 3 | ucr85:Trace | 1/1 | 0.8219 | 0.7945 | 0.8100 |
| A1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.5000 | 0.3331 | 0.4996 |
| A1 | 3 | ucr85:Two_Patterns | 1/1 | 0.7982 | 0.7977 | 0.7983 |
| A1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.3127 | 0.3265 | 0.5423 |
| A1 | 3 | ucr85:Worms | 1/1 | 0.2560 | 0.2302 | 0.2707 |
| A1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.5200 | 0.4683 | 0.5801 |
| A1 | 3 | ucr85:synthetic_control | 1/1 | 0.9233 | 0.9221 | 0.9233 |
| A1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7377 | 0.7254 | 0.7376 |
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
| F1 | 3 | raman34:head_neck_cancer | 1/1 | 0.5117 | 0.4665 | 0.6306 |
| F1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.9300 | 0.9300 | 0.9300 |
| F1 | 3 | raman34:marine_pathogens | 2/2 | 0.9890 | 0.9891 | 0.9895 |
| F1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9732 | 0.9750 | 0.9729 |
| F1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.2571 | 0.2222 | 0.2941 |
| F1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8237 | 0.8189 | 0.8193 |
| F1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8200 | 0.8199 | 0.8200 |
| F1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.8361 | 0.8441 | 0.8800 |
| F1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.5637 | 0.5579 | 0.6905 |
| F1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8274 | 0.8269 | 0.8269 |
| F1 | 3 | tabmini25:backache | 1/1 | 0.4968 | 0.4611 | 0.8556 |
| F1 | 3 | tabmini25:biomed | 1/1 | 0.8376 | 0.8523 | 0.8708 |
| F1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5156 | 0.4843 | 0.6818 |
| F1 | 3 | tabmini25:bupa | 1/1 | 0.6391 | 0.6365 | 0.6377 |
| F1 | 3 | tabmini25:corral | 1/1 | 0.9071 | 0.9052 | 0.9062 |
| F1 | 3 | tabmini25:haberman | 1/1 | 0.5057 | 0.4453 | 0.7320 |
| F1 | 3 | tabmini25:hepatitis | 1/1 | 0.6278 | 0.6485 | 0.8129 |
| F1 | 3 | tabmini25:irish | 1/1 | 0.7529 | 0.7546 | 0.7600 |
| F1 | 3 | tabmini25:labor | 1/1 | 0.8095 | 0.8199 | 0.8421 |
| F1 | 3 | tabmini25:lupus | 1/1 | 0.7374 | 0.7421 | 0.7586 |
| F1 | 3 | tabmini25:molecular_biology_promoters | 1/1 | 0.7736 | 0.7733 | 0.7736 |
| F1 | 3 | tabmini25:mux6 | 1/1 | 0.6719 | 0.6679 | 0.6719 |
| F1 | 3 | tabmini25:postoperative_patient_data | 1/1 | 0.5000 | 0.4211 | 0.7273 |
| F1 | 3 | tabmini25:prnn_synth | 1/1 | 0.8440 | 0.8440 | 0.8440 |
| F1 | 3 | tabmini25:saheart | 1/1 | 0.6567 | 0.6637 | 0.7165 |
| F1 | 3 | ucr85:Adiac | 1/1 | 0.6196 | 0.5589 | 0.6036 |
| F1 | 3 | ucr85:ArrowHead | 1/1 | 0.7346 | 0.7268 | 0.7257 |
| F1 | 3 | ucr85:CBF | 1/1 | 0.8917 | 0.8909 | 0.8911 |
| F1 | 3 | ucr85:Car | 1/1 | 0.8526 | 0.8515 | 0.8500 |
| F1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.8666 | 0.8852 | 0.8953 |
| F1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.4154 | 0.4144 | 0.4145 |
| F1 | 3 | ucr85:Computers | 1/1 | 0.5080 | 0.5051 | 0.5080 |
| F1 | 3 | ucr85:Cricket | 3/3 | 0.3869 | 0.3690 | 0.3795 |
| F1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5183 | 0.4967 | 0.7600 |
| F1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7072 | 0.7159 | 0.7517 |
| F1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4629 | 0.4288 | 0.7725 |
| F1 | 3 | ucr85:ECG200 | 1/1 | 0.7691 | 0.7707 | 0.7900 |
| F1 | 3 | ucr85:ECG5000 | 1/1 | 0.4227 | 0.4255 | 0.9258 |
| F1 | 3 | ucr85:Earthquakes | 1/1 | 0.5000 | 0.4505 | 0.8199 |
| F1 | 3 | ucr85:ElectricDevices | 1/1 | 0.3667 | 0.3398 | 0.4907 |
| F1 | 3 | ucr85:FISH | 1/1 | 0.8541 | 0.8503 | 0.8514 |
| F1 | 3 | ucr85:FaceAll | 1/1 | 0.7973 | 0.7340 | 0.7118 |
| F1 | 3 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| F1 | 3 | ucr85:FacesUCR | 1/1 | 0.6417 | 0.6448 | 0.7244 |
| F1 | 3 | ucr85:FordA | 1/1 | 0.5000 | 0.3389 | 0.5126 |
| F1 | 3 | ucr85:FordB | 1/1 | 0.4862 | 0.4861 | 0.4862 |
| F1 | 3 | ucr85:Gun_Point | 1/1 | 0.8994 | 0.8996 | 0.9000 |
| F1 | 3 | ucr85:Ham | 1/1 | 0.6187 | 0.6187 | 0.6190 |
| F1 | 3 | ucr85:HandOutlines | 1/1 | 0.8266 | 0.8372 | 0.8550 |
| F1 | 3 | ucr85:Haptics | 1/1 | 0.4134 | 0.3802 | 0.4188 |
| F1 | 3 | ucr85:Herring | 1/1 | 0.5000 | 0.3725 | 0.5938 |
| F1 | 3 | ucr85:InlineSkate | 1/1 | 0.2151 | 0.1809 | 0.2364 |
| F1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5990 | 0.5903 | 0.5990 |
| F1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9611 | 0.9611 | 0.9611 |
| F1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.3867 | 0.3861 | 0.3867 |
| F1 | 3 | ucr85:Lighting2 | 1/1 | 0.6634 | 0.6420 | 0.6885 |
| F1 | 3 | ucr85:Lighting7 | 1/1 | 0.5937 | 0.5534 | 0.6164 |
| F1 | 3 | ucr85:MALLAT | 1/1 | 0.8617 | 0.8559 | 0.8606 |
| F1 | 3 | ucr85:MedicalImages | 1/1 | 0.3637 | 0.3995 | 0.6039 |
| F1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.4174 | 0.2978 | 0.4225 |
| F1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5951 | 0.5482 | 0.5483 |
| F1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3309 | 0.2781 | 0.5965 |
| F1 | 3 | ucr85:MoteStrain | 1/1 | 0.8685 | 0.8683 | 0.8690 |
| F1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9296 | 0.9280 | 0.9310 |
| F1 | 3 | ucr85:OSULeaf | 1/1 | 0.3952 | 0.3520 | 0.4091 |
| F1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.5846 | 0.5612 | 0.6643 |
| F1 | 3 | ucr85:Plane | 1/1 | 0.9911 | 0.9908 | 0.9905 |
| F1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.7235 | 0.7151 | 0.8244 |
| F1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.7980 | 0.8190 | 0.8557 |
| F1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4730 | 0.4542 | 0.7975 |
| F1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.3600 | 0.3409 | 0.3600 |
| F1 | 3 | ucr85:ScreenType | 1/1 | 0.3787 | 0.3768 | 0.3787 |
| F1 | 3 | ucr85:ShapeletSim | 1/1 | 0.5389 | 0.5052 | 0.5389 |
| F1 | 3 | ucr85:ShapesAll | 1/1 | 0.6667 | 0.6307 | 0.6667 |
| F1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.4640 | 0.4586 | 0.4640 |
| F1 | 3 | ucr85:StarLightCurves | 1/1 | 0.8842 | 0.8886 | 0.9246 |
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
| P0 | 1 | biology5:colon | 1/1 | 0.7784 | 0.7840 | 0.8065 |
| P0 | 1 | raman34:cancer_cell_media | 3/3 | 0.9528 | 0.9529 | 0.9529 |
| P0 | 1 | raman34:chlorinated_samples | 1/1 | 0.9247 | 0.9219 | 0.9304 |
| P0 | 1 | raman34:comfile_stroke | 1/1 | 0.9000 | 0.9000 | 0.9000 |
| P0 | 1 | raman34:cspp_serum_metabolites | 1/1 | 0.9333 | 0.9327 | 0.9333 |
| P0 | 1 | raman34:diabetes_skin | 4/4 | 0.4798 | 0.4618 | 0.4750 |
| P0 | 1 | raman34:hair_dyes_sers | 1/1 | 0.9983 | 0.9980 | 0.9982 |
| P0 | 1 | raman34:head_neck_cancer | 1/1 | 0.4817 | 0.4597 | 0.5856 |
| P0 | 1 | raman34:locust_phase_hemolymph | 1/1 | 0.8700 | 0.8700 | 0.8700 |
| P0 | 1 | raman34:marine_pathogens | 2/2 | 0.9894 | 0.9895 | 0.9899 |
| P0 | 1 | raman34:pharmaceutical_ingredients | 1/1 | 0.9979 | 0.9979 | 0.9980 |
| P0 | 1 | raman34:serum_alzheimer_disease | 1/1 | 0.2857 | 0.1818 | 0.3529 |
| P0 | 1 | tabmini25:analcatdata_asbestos | 1/1 | 0.8211 | 0.8183 | 0.8193 |
| P0 | 1 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8600 | 0.8599 | 0.8600 |
| P0 | 1 | tabmini25:analcatdata_creditscore | 1/1 | 0.9932 | 0.9875 | 0.9900 |
| P0 | 1 | tabmini25:analcatdata_fraud | 1/1 | 0.6101 | 0.6101 | 0.6667 |
| P0 | 1 | tabmini25:analcatdata_japansolvent | 1/1 | 0.7830 | 0.7819 | 0.7885 |
| P0 | 1 | tabmini25:backache | 1/1 | 0.4948 | 0.4908 | 0.7944 |
| P0 | 1 | tabmini25:biomed | 1/1 | 0.8398 | 0.8496 | 0.8660 |
| P0 | 1 | tabmini25:breast_cancer | 1/1 | 0.5754 | 0.5728 | 0.7133 |
| P0 | 1 | tabmini25:bupa | 1/1 | 0.6209 | 0.6202 | 0.6203 |
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
| P0 | 1 | tabmini25:saheart | 1/1 | 0.6720 | 0.6784 | 0.7229 |
| P0 | 1 | ucr85:Adiac | 1/1 | 0.6196 | 0.5589 | 0.6036 |
| P0 | 1 | ucr85:ArrowHead | 1/1 | 0.5817 | 0.5818 | 0.5829 |
| P0 | 1 | ucr85:CBF | 1/1 | 0.8496 | 0.8478 | 0.8489 |
| P0 | 1 | ucr85:Car | 1/1 | 0.7629 | 0.7534 | 0.7500 |
| P0 | 1 | ucr85:ChlorineConcentration | 1/1 | 0.8630 | 0.8732 | 0.8888 |
| P0 | 1 | ucr85:CinC_ECG_torso | 1/1 | 0.5901 | 0.5865 | 0.5899 |
| P0 | 1 | ucr85:Computers | 1/1 | 0.7720 | 0.7714 | 0.7720 |
| P0 | 1 | ucr85:Cricket | 3/3 | 0.7927 | 0.7886 | 0.7932 |
| P0 | 1 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5585 | 0.5310 | 0.8025 |
| P0 | 1 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7676 | 0.7676 | 0.7833 |
| P0 | 1 | ucr85:DistalPhalanxTW | 1/1 | 0.4654 | 0.4310 | 0.7450 |
| P0 | 1 | ucr85:ECG200 | 1/1 | 0.8420 | 0.8462 | 0.8600 |
| P0 | 1 | ucr85:ECG5000 | 1/1 | 0.4545 | 0.4663 | 0.9236 |
| P0 | 1 | ucr85:Earthquakes | 1/1 | 0.5875 | 0.5256 | 0.5994 |
| P0 | 1 | ucr85:ElectricDevices | 1/1 | 0.6607 | 0.6657 | 0.7421 |
| P0 | 1 | ucr85:FISH | 1/1 | 0.8732 | 0.8702 | 0.8743 |
| P0 | 1 | ucr85:FaceAll | 1/1 | 0.9095 | 0.8582 | 0.8538 |
| P0 | 1 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| P0 | 1 | ucr85:FacesUCR | 1/1 | 0.7793 | 0.7825 | 0.8346 |
| P0 | 1 | ucr85:FordA | 1/1 | 0.9036 | 0.9028 | 0.9028 |
| P0 | 1 | ucr85:FordB | 1/1 | 0.9275 | 0.9276 | 0.9277 |
| P0 | 1 | ucr85:Gun_Point | 1/1 | 0.8535 | 0.8533 | 0.8533 |
| P0 | 1 | ucr85:Ham | 1/1 | 0.6656 | 0.6656 | 0.6667 |
| P0 | 1 | ucr85:HandOutlines | 1/1 | 0.8190 | 0.8346 | 0.8560 |
| P0 | 1 | ucr85:Haptics | 1/1 | 0.4450 | 0.4032 | 0.4481 |
| P0 | 1 | ucr85:Herring | 1/1 | 0.5020 | 0.4000 | 0.4375 |
| P0 | 1 | ucr85:InlineSkate | 1/1 | 0.3473 | 0.3386 | 0.3473 |
| P0 | 1 | ucr85:InsectWingbeatSound | 1/1 | 0.6303 | 0.6282 | 0.6303 |
| P0 | 1 | ucr85:ItalyPowerDemand | 1/1 | 0.9290 | 0.9291 | 0.9291 |
| P0 | 1 | ucr85:LargeKitchenAppliances | 1/1 | 0.8453 | 0.8448 | 0.8453 |
| P0 | 1 | ucr85:Lighting2 | 1/1 | 0.6921 | 0.6909 | 0.7049 |
| P0 | 1 | ucr85:Lighting7 | 1/1 | 0.6279 | 0.5868 | 0.6301 |
| P0 | 1 | ucr85:MALLAT | 1/1 | 0.8970 | 0.8898 | 0.8968 |
| P0 | 1 | ucr85:MedicalImages | 1/1 | 0.5037 | 0.5357 | 0.6934 |
| P0 | 1 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6664 | 0.6522 | 0.7950 |
| P0 | 1 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5874 | 0.5297 | 0.5300 |
| P0 | 1 | ucr85:MiddlePhalanxTW | 1/1 | 0.3751 | 0.3510 | 0.6190 |
| P0 | 1 | ucr85:MoteStrain | 1/1 | 0.7944 | 0.7948 | 0.7963 |
| P0 | 1 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9296 | 0.9280 | 0.9310 |
| P0 | 1 | ucr85:OSULeaf | 1/1 | 0.9705 | 0.9747 | 0.9752 |
| P0 | 1 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7826 | 0.7790 | 0.7879 |
| P0 | 1 | ucr85:Plane | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| P0 | 1 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8111 | 0.7653 | 0.8341 |
| P0 | 1 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8633 | 0.8762 | 0.8969 |
| P0 | 1 | ucr85:ProximalPhalanxTW | 1/1 | 0.4953 | 0.4914 | 0.8050 |
| P0 | 1 | ucr85:RefrigerationDevices | 1/1 | 0.5600 | 0.5598 | 0.5600 |
| P0 | 1 | ucr85:ScreenType | 1/1 | 0.5013 | 0.4988 | 0.5013 |
| P0 | 1 | ucr85:ShapeletSim | 1/1 | 0.8389 | 0.8380 | 0.8389 |
| P0 | 1 | ucr85:ShapesAll | 1/1 | 0.8533 | 0.8484 | 0.8533 |
| P0 | 1 | ucr85:SmallKitchenAppliances | 1/1 | 0.7387 | 0.7366 | 0.7387 |
| P0 | 1 | ucr85:StarLightCurves | 1/1 | 0.9482 | 0.9361 | 0.9541 |
| P0 | 1 | ucr85:Strawberry | 1/1 | 0.9721 | 0.9683 | 0.9706 |
| P0 | 1 | ucr85:SwedishLeaf | 1/1 | 0.9427 | 0.9416 | 0.9408 |
| P0 | 1 | ucr85:Symbols | 1/1 | 0.8471 | 0.8445 | 0.8452 |
| P0 | 1 | ucr85:ToeSegmentation1 | 1/1 | 0.9153 | 0.9123 | 0.9123 |
| P0 | 1 | ucr85:ToeSegmentation2 | 1/1 | 0.8420 | 0.8365 | 0.9000 |
| P0 | 1 | ucr85:Trace | 1/1 | 0.9911 | 0.9890 | 0.9900 |
| P0 | 1 | ucr85:TwoLeadECG | 1/1 | 0.6587 | 0.6412 | 0.6585 |
| P0 | 1 | ucr85:Two_Patterns | 1/1 | 0.9988 | 0.9988 | 0.9988 |
| P0 | 1 | ucr85:WordsSynonyms | 1/1 | 0.3144 | 0.2910 | 0.5345 |
| P0 | 1 | ucr85:Worms | 1/1 | 0.4976 | 0.4947 | 0.5470 |
| P0 | 1 | ucr85:WormsTwoClass | 1/1 | 0.5012 | 0.4848 | 0.5414 |
| P0 | 1 | ucr85:synthetic_control | 1/1 | 0.9867 | 0.9867 | 0.9867 |
| P0 | 1 | ucr85:uWaveGestureLibrary | 4/4 | 0.7888 | 0.7843 | 0.7885 |
| P0 | 1 | ucr85:wafer | 1/1 | 0.7378 | 0.8005 | 0.9401 |
| P0 | 1 | ucr85:yoga | 1/1 | 0.8238 | 0.8244 | 0.8257 |
| P1 | 3 | biology5:ALLAML | 1/1 | 0.9268 | 0.9111 | 0.9167 |
| P1 | 3 | biology5:CLL_SUB_111 | 1/1 | 0.8195 | 0.8195 | 0.7568 |
| P1 | 3 | biology5:Carcinom | 1/1 | 0.8399 | 0.8540 | 0.8793 |
| P1 | 3 | biology5:GLIOMA | 1/1 | 0.7702 | 0.7797 | 0.8000 |
| P1 | 3 | biology5:colon | 1/1 | 0.8364 | 0.8398 | 0.8548 |
| P1 | 3 | raman34:cancer_cell_media | 3/3 | 0.9563 | 0.9562 | 0.9561 |
| P1 | 3 | raman34:chlorinated_samples | 1/1 | 0.9181 | 0.9168 | 0.9261 |
| P1 | 3 | raman34:comfile_stroke | 1/1 | 0.9000 | 0.9000 | 0.9000 |
| P1 | 3 | raman34:cspp_serum_metabolites | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| P1 | 3 | raman34:diabetes_skin | 4/4 | 0.4823 | 0.4616 | 0.4750 |
| P1 | 3 | raman34:hair_dyes_sers | 1/1 | 0.9967 | 0.9970 | 0.9977 |
| P1 | 3 | raman34:head_neck_cancer | 1/1 | 0.4927 | 0.4886 | 0.5766 |
| P1 | 3 | raman34:locust_phase_hemolymph | 1/1 | 0.8700 | 0.8700 | 0.8700 |
| P1 | 3 | raman34:marine_pathogens | 2/2 | 0.9888 | 0.9890 | 0.9894 |
| P1 | 3 | raman34:pharmaceutical_ingredients | 1/1 | 0.9982 | 0.9982 | 0.9983 |
| P1 | 3 | raman34:serum_alzheimer_disease | 1/1 | 0.3714 | 0.3088 | 0.4118 |
| P1 | 3 | tabmini25:analcatdata_asbestos | 1/1 | 0.8211 | 0.8183 | 0.8193 |
| P1 | 3 | tabmini25:analcatdata_bankruptcy | 1/1 | 0.8600 | 0.8599 | 0.8600 |
| P1 | 3 | tabmini25:analcatdata_creditscore | 1/1 | 0.9932 | 0.9875 | 0.9900 |
| P1 | 3 | tabmini25:analcatdata_fraud | 1/1 | 0.6101 | 0.6101 | 0.6667 |
| P1 | 3 | tabmini25:analcatdata_japansolvent | 1/1 | 0.8244 | 0.8253 | 0.8269 |
| P1 | 3 | tabmini25:backache | 1/1 | 0.4974 | 0.4829 | 0.8278 |
| P1 | 3 | tabmini25:biomed | 1/1 | 0.8398 | 0.8496 | 0.8660 |
| P1 | 3 | tabmini25:breast_cancer | 1/1 | 0.5754 | 0.5728 | 0.7133 |
| P1 | 3 | tabmini25:bupa | 1/1 | 0.5983 | 0.5962 | 0.5971 |
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
| P1 | 3 | tabmini25:saheart | 1/1 | 0.6567 | 0.6637 | 0.7165 |
| P1 | 3 | ucr85:Adiac | 1/1 | 0.7791 | 0.7577 | 0.7647 |
| P1 | 3 | ucr85:ArrowHead | 1/1 | 0.7327 | 0.7216 | 0.7200 |
| P1 | 3 | ucr85:CBF | 1/1 | 0.8523 | 0.8506 | 0.8522 |
| P1 | 3 | ucr85:Car | 1/1 | 0.6887 | 0.6849 | 0.6833 |
| P1 | 3 | ucr85:ChlorineConcentration | 1/1 | 0.8459 | 0.8618 | 0.8836 |
| P1 | 3 | ucr85:CinC_ECG_torso | 1/1 | 0.6972 | 0.6923 | 0.6964 |
| P1 | 3 | ucr85:Computers | 1/1 | 0.7720 | 0.7714 | 0.7720 |
| P1 | 3 | ucr85:Cricket | 3/3 | 0.7952 | 0.7882 | 0.7957 |
| P1 | 3 | ucr85:DistalPhalanxOutlineAgeGroup | 1/1 | 0.5805 | 0.5693 | 0.7850 |
| P1 | 3 | ucr85:DistalPhalanxOutlineCorrect | 1/1 | 0.7676 | 0.7676 | 0.7833 |
| P1 | 3 | ucr85:DistalPhalanxTW | 1/1 | 0.4890 | 0.4618 | 0.7225 |
| P1 | 3 | ucr85:ECG200 | 1/1 | 0.8420 | 0.8462 | 0.8600 |
| P1 | 3 | ucr85:ECG5000 | 1/1 | 0.4545 | 0.4663 | 0.9236 |
| P1 | 3 | ucr85:Earthquakes | 1/1 | 0.6482 | 0.5345 | 0.5776 |
| P1 | 3 | ucr85:ElectricDevices | 1/1 | 0.6461 | 0.6516 | 0.7347 |
| P1 | 3 | ucr85:FISH | 1/1 | 0.8501 | 0.8484 | 0.8514 |
| P1 | 3 | ucr85:FaceAll | 1/1 | 0.9260 | 0.8816 | 0.8911 |
| P1 | 3 | ucr85:FaceFour | 1/1 | 0.8807 | 0.8773 | 0.8750 |
| P1 | 3 | ucr85:FacesUCR | 1/1 | 0.7242 | 0.7235 | 0.7946 |
| P1 | 3 | ucr85:FordA | 1/1 | 0.9361 | 0.9353 | 0.9353 |
| P1 | 3 | ucr85:FordB | 1/1 | 0.9275 | 0.9276 | 0.9277 |
| P1 | 3 | ucr85:Gun_Point | 1/1 | 0.8535 | 0.8533 | 0.8533 |
| P1 | 3 | ucr85:Ham | 1/1 | 0.6656 | 0.6656 | 0.6667 |
| P1 | 3 | ucr85:HandOutlines | 1/1 | 0.7961 | 0.8143 | 0.8420 |
| P1 | 3 | ucr85:Haptics | 1/1 | 0.4450 | 0.4032 | 0.4481 |
| P1 | 3 | ucr85:Herring | 1/1 | 0.4737 | 0.3600 | 0.5625 |
| P1 | 3 | ucr85:InlineSkate | 1/1 | 0.4345 | 0.4118 | 0.4073 |
| P1 | 3 | ucr85:InsectWingbeatSound | 1/1 | 0.5965 | 0.5864 | 0.5965 |
| P1 | 3 | ucr85:ItalyPowerDemand | 1/1 | 0.9153 | 0.9153 | 0.9155 |
| P1 | 3 | ucr85:LargeKitchenAppliances | 1/1 | 0.8453 | 0.8448 | 0.8453 |
| P1 | 3 | ucr85:Lighting2 | 1/1 | 0.6564 | 0.6510 | 0.6721 |
| P1 | 3 | ucr85:Lighting7 | 1/1 | 0.4924 | 0.4768 | 0.5068 |
| P1 | 3 | ucr85:MALLAT | 1/1 | 0.8970 | 0.8898 | 0.8968 |
| P1 | 3 | ucr85:MedicalImages | 1/1 | 0.5752 | 0.5878 | 0.6908 |
| P1 | 3 | ucr85:MiddlePhalanxOutlineAgeGroup | 1/1 | 0.6664 | 0.6522 | 0.7950 |
| P1 | 3 | ucr85:MiddlePhalanxOutlineCorrect | 1/1 | 0.5874 | 0.5297 | 0.5300 |
| P1 | 3 | ucr85:MiddlePhalanxTW | 1/1 | 0.3309 | 0.2781 | 0.5965 |
| P1 | 3 | ucr85:MoteStrain | 1/1 | 0.8052 | 0.8025 | 0.8027 |
| P1 | 3 | ucr85:NonInvasiveFatalECG_Thorax | 2/2 | 0.9296 | 0.9280 | 0.9310 |
| P1 | 3 | ucr85:OSULeaf | 1/1 | 0.9651 | 0.9651 | 0.9669 |
| P1 | 3 | ucr85:PhalangesOutlinesCorrect | 1/1 | 0.7805 | 0.7823 | 0.7949 |
| P1 | 3 | ucr85:Plane | 1/1 | 1.0000 | 1.0000 | 1.0000 |
| P1 | 3 | ucr85:ProximalPhalanxOutlineAgeGroup | 1/1 | 0.8082 | 0.7574 | 0.8293 |
| P1 | 3 | ucr85:ProximalPhalanxOutlineCorrect | 1/1 | 0.8545 | 0.8736 | 0.8969 |
| P1 | 3 | ucr85:ProximalPhalanxTW | 1/1 | 0.4914 | 0.4866 | 0.7900 |
| P1 | 3 | ucr85:RefrigerationDevices | 1/1 | 0.5600 | 0.5598 | 0.5600 |
| P1 | 3 | ucr85:ScreenType | 1/1 | 0.5013 | 0.4988 | 0.5013 |
| P1 | 3 | ucr85:ShapeletSim | 1/1 | 0.8389 | 0.8380 | 0.8389 |
| P1 | 3 | ucr85:ShapesAll | 1/1 | 0.8533 | 0.8484 | 0.8533 |
| P1 | 3 | ucr85:SmallKitchenAppliances | 1/1 | 0.7387 | 0.7366 | 0.7387 |
| P1 | 3 | ucr85:StarLightCurves | 1/1 | 0.9482 | 0.9361 | 0.9541 |
| P1 | 3 | ucr85:Strawberry | 1/1 | 0.9338 | 0.9281 | 0.9331 |
| P1 | 3 | ucr85:SwedishLeaf | 1/1 | 0.9500 | 0.9491 | 0.9488 |
| P1 | 3 | ucr85:Symbols | 1/1 | 0.8471 | 0.8445 | 0.8452 |
| P1 | 3 | ucr85:ToeSegmentation1 | 1/1 | 0.9153 | 0.9123 | 0.9123 |
| P1 | 3 | ucr85:ToeSegmentation2 | 1/1 | 0.8420 | 0.8365 | 0.9000 |
| P1 | 3 | ucr85:Trace | 1/1 | 0.9911 | 0.9890 | 0.9900 |
| P1 | 3 | ucr85:TwoLeadECG | 1/1 | 0.6929 | 0.6798 | 0.6927 |
| P1 | 3 | ucr85:Two_Patterns | 1/1 | 0.9988 | 0.9988 | 0.9988 |
| P1 | 3 | ucr85:WordsSynonyms | 1/1 | 0.3349 | 0.3095 | 0.5219 |
| P1 | 3 | ucr85:Worms | 1/1 | 0.4976 | 0.4947 | 0.5470 |
| P1 | 3 | ucr85:WormsTwoClass | 1/1 | 0.7192 | 0.7239 | 0.7459 |
| P1 | 3 | ucr85:synthetic_control | 1/1 | 0.9867 | 0.9867 | 0.9867 |
| P1 | 3 | ucr85:uWaveGestureLibrary | 4/4 | 0.7854 | 0.7805 | 0.7852 |
| P1 | 3 | ucr85:wafer | 1/1 | 0.8100 | 0.8494 | 0.9487 |
| P1 | 3 | ucr85:yoga | 1/1 | 0.8238 | 0.8244 | 0.8257 |

## 开发任务确定的固定模型

- budget=1：模型 svm，开发任务平均BA 0.8028；候选 logistic_regression, pca_lda, pca_mlp, pls_da, random_forest, svm, xgboost。
- budget=3：模型 svm，开发任务平均BA 0.8000；候选 logistic_regression, pca_lda, pca_mlp, pls_da, random_forest, svm, xgboost。

## 配对差

差值为左侧方法减右侧方法；95% 区间按 family_id 聚类 bootstrap；tie 阈值为 1e-12。

| 比较 | 指标 | 任务数 | 平均差 | 95% CI | Win/Tie/Loss |
|---|---|---:|---:|---|---:|
| A0-R0 | balanced_accuracy | 114 | -0.0054 | [-0.0301, 0.0156] | 73/0/41 |
| A0-R0 | accuracy | 114 | -0.0078 | [-0.0317, 0.0147] | 71/2/41 |
| A0-R0 | macro_f1 | 114 | -0.0049 | [-0.0325, 0.0211] | 72/0/42 |
| A1-R1 | balanced_accuracy | 114 | -0.0013 | [-0.0197, 0.0164] | 61/1/52 |
| A1-R1 | accuracy | 114 | -0.0013 | [-0.0182, 0.0154] | 62/2/50 |
| A1-R1 | macro_f1 | 114 | -0.0001 | [-0.0186, 0.0190] | 60/0/54 |
| F0-R0 | balanced_accuracy | 114 | 0.0029 | [-0.0196, 0.0254] | 59/0/55 |
| F0-R0 | accuracy | 114 | 0.0059 | [-0.0156, 0.0273] | 65/1/48 |
| F0-R0 | macro_f1 | 114 | 0.0026 | [-0.0206, 0.0256] | 59/0/55 |
| F0-A0 | balanced_accuracy | 114 | 0.0083 | [-0.0120, 0.0318] | 48/9/57 |
| F0-A0 | accuracy | 114 | 0.0136 | [-0.0065, 0.0360] | 53/11/50 |
| F0-A0 | macro_f1 | 114 | 0.0075 | [-0.0147, 0.0310] | 48/9/57 |
| F0-P0 | balanced_accuracy | 114 | -0.0808 | [-0.1112, -0.0524] | 24/9/81 |
| F0-P0 | accuracy | 114 | -0.0667 | [-0.0995, -0.0358] | 34/10/70 |
| F0-P0 | macro_f1 | 114 | -0.0870 | [-0.1209, -0.0584] | 27/8/79 |
| F1-R1 | balanced_accuracy | 114 | -0.0134 | [-0.0332, 0.0072] | 53/0/61 |
| F1-R1 | accuracy | 114 | -0.0077 | [-0.0291, 0.0122] | 60/2/52 |
| F1-R1 | macro_f1 | 114 | -0.0161 | [-0.0384, 0.0052] | 52/0/62 |
| F1-A1 | balanced_accuracy | 114 | -0.0121 | [-0.0275, 0.0046] | 47/15/52 |
| F1-A1 | accuracy | 114 | -0.0064 | [-0.0216, 0.0094] | 50/18/46 |
| F1-A1 | macro_f1 | 114 | -0.0160 | [-0.0351, 0.0042] | 49/15/50 |
| F1-P1 | balanced_accuracy | 114 | -0.0850 | [-0.1155, -0.0556] | 32/8/74 |
| F1-P1 | accuracy | 114 | -0.0696 | [-0.1020, -0.0400] | 37/13/64 |
| F1-P1 | macro_f1 | 114 | -0.0913 | [-0.1261, -0.0604] | 30/8/76 |

## 平均秩

| 预算 | 指标 | 方法 | 平均秩 | 任务数 |
|---:|---|---|---:|---:|
| 1 | balanced_accuracy | R0 | 3.009 | 114 |
| 1 | balanced_accuracy | A0 | 2.491 | 114 |
| 1 | balanced_accuracy | F0 | 2.772 | 114 |
| 1 | balanced_accuracy | P0 | 1.728 | 114 |
| 1 | accuracy | R0 | 3.018 | 114 |
| 1 | accuracy | A0 | 2.522 | 114 |
| 1 | accuracy | F0 | 2.570 | 114 |
| 1 | accuracy | P0 | 1.890 | 114 |
| 1 | macro_f1 | R0 | 2.991 | 114 |
| 1 | macro_f1 | A0 | 2.491 | 114 |
| 1 | macro_f1 | F0 | 2.750 | 114 |
| 1 | macro_f1 | P0 | 1.768 | 114 |
| 3 | balanced_accuracy | R1 | 2.833 | 114 |
| 3 | balanced_accuracy | A1 | 2.667 | 114 |
| 3 | balanced_accuracy | F1 | 2.741 | 114 |
| 3 | balanced_accuracy | P1 | 1.759 | 114 |
| 3 | accuracy | R1 | 2.825 | 114 |
| 3 | accuracy | A1 | 2.632 | 114 |
| 3 | accuracy | F1 | 2.566 | 114 |
| 3 | accuracy | P1 | 1.978 | 114 |
| 3 | macro_f1 | R1 | 2.798 | 114 |
| 3 | macro_f1 | A1 | 2.689 | 114 |
| 3 | macro_f1 | F1 | 2.750 | 114 |
| 3 | macro_f1 | P1 | 1.763 | 114 |

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
| development | 1 | 305 | 515.000 | 4274.454 | 0.000 | 0.000 | 515.000 | 4274.454 | 62290.000 | 0.000 | 0.000 |
| development | 3 | 305 | 596.000 | 6923.663 | 466.000 | 7.750 | 1062.000 | 8999.901 | 62221.000 | 0.000 | 0.000 |
| locked | 1 | 2890 | 4570.000 | 145546.584 | 0.000 | 0.000 | 4570.000 | 145546.584 | 620104.000 | 0.000 | 0.000 |
| locked | 3 | 2890 | 6221.000 | 262969.079 | 3915.000 | 123.530 | 10136.000 | 353106.841 | 619863.000 | 0.000 | 0.000 |

### 任务实际墙钟时间

task_wall_seconds 来自 driver_state 的任务时钟；driver_state 未提供时保留为未知，不以 Fit duration sum 代替。

| 任务 | 角色 | task_wall_seconds | 云端核验时间 |
|---|---|---:|---|
| Beef | development | 1106.918 | 1791651930.892415 |
| BeetleFly | development | 1107.142 | 1791651931.2753468 |
| BirdChicken | development | 1107.357 | 1791651931.6585262 |
| Coffee | development | 1107.435 | 1791651931.9032443 |
| ECGFiveDays | development | 1107.625 | 1791651932.2902792 |
| Meat | development | 1107.874 | 1791651932.7087715 |
| OliveOil | development | 1108.112 | 1791651933.1143365 |
| SonyAIBORobotSurface | development | 1108.208 | 1791651933.36764 |
| SonyAIBORobotSurfaceII | development | 1108.460 | 1791651933.7845469 |
| Wine | development | 1108.721 | 1791651934.200357 |
| analcatdata_aids | development | 1108.805 | 1791651934.4625828 |
| appendicitis | development | 1109.009 | 1791651934.8470476 |
| parity5 | development | 1109.238 | 1791651935.2516992 |
| prnn_crabs | development | 1109.309 | 1791651935.5099754 |
| glass2 | development | 1109.517 | 1791651935.9025297 |
| ALLAML | locked | 28131.407 | 1791680127.411218 |
| CLL_SUB_111 | locked | 28145.654 | 1791680141.8440328 |
| Carcinom | locked | 28160.162 | 1791680156.544665 |
| GLIOMA | locked | 28174.045 | 1791680170.611244 |
| colon | locked | 28188.766 | 1791680185.5150034 |
| cancer_cell_(cooh)2 | locked | 28202.348 | 1791680199.2983634 |
| cancer_cell_cooh | locked | 28216.714 | 1791680213.873127 |
| cancer_cell_nh2 | locked | 28230.806 | 1791680228.162365 |
| chlorinated_samples | locked | 28245.199 | 1791680242.743893 |
| comfile_stroke | locked | 28259.195 | 1791680256.9465225 |
| cspp_serum_metabolites | locked | 28273.531 | 1791680271.486147 |
| diabetes_skin_ear_lobe | locked | 28288.326 | 1791680286.5196435 |
| diabetes_skin_inner_arm | locked | 28304.560 | 1791680302.9564803 |
| diabetes_skin_thumbnail | locked | 28320.930 | 1791680319.5299766 |
| diabetes_skin_vein | locked | 28335.593 | 1791680334.4031718 |
| hair_dyes_sers | locked | 28350.061 | 1791680349.0748088 |
| head_neck_cancer | locked | 28365.927 | 1791680365.142247 |
| locust_phase_hemolymph | locked | 28381.640 | 1791680381.0564184 |
| marine_pathogens | locked | 28397.121 | 1791680396.7515187 |
| marine_pathogens_binary | locked | 28411.844 | 1791680411.6756892 |
| pharmaceutical_ingredients | locked | 28426.677 | 1791680426.7213032 |
| serum_alzheimer_disease | locked | 28441.286 | 1791680441.519433 |
| analcatdata_asbestos | locked | 28455.659 | 1791680456.0730312 |
| analcatdata_bankruptcy | locked | 28470.706 | 1791680471.294786 |
| analcatdata_creditscore | locked | 28484.641 | 1791680485.4118807 |
| analcatdata_fraud | locked | 28499.501 | 1791680500.4414568 |
| analcatdata_japansolvent | locked | 28513.835 | 1791680514.9679658 |
| backache | locked | 28527.830 | 1791680529.1460984 |
| biomed | locked | 28542.897 | 1791680544.3953762 |
| breast_cancer | locked | 28556.808 | 1791680558.4857934 |
| bupa | locked | 28571.346 | 1791680573.2149477 |
| corral | locked | 28586.332 | 1791680588.3930116 |
| haberman | locked | 28600.269 | 1791680602.5227535 |
| hepatitis | locked | 28614.665 | 1791680617.1154273 |
| irish | locked | 28629.516 | 1791680632.1468742 |
| labor | locked | 28643.876 | 1791680646.6876953 |
| lupus | locked | 28658.353 | 1791680661.3525305 |
| molecular_biology_promoters | locked | 28672.551 | 1791680675.7321281 |
| mux6 | locked | 28687.159 | 1791680690.5125632 |
| postoperative_patient_data | locked | 28701.942 | 1791680705.4753308 |
| prnn_synth | locked | 28715.924 | 1791680719.6382072 |
| saheart | locked | 28731.051 | 1791680734.9624128 |
| Adiac | locked | 28745.081 | 1791680749.167228 |
| ArrowHead | locked | 28759.333 | 1791680763.6133332 |
| CBF | locked | 28774.090 | 1791680778.5403948 |
| Car | locked | 28789.078 | 1791680793.690265 |
| ChlorineConcentration | locked | 28802.940 | 1791680807.7112093 |
| CinC_ECG_torso | locked | 28817.495 | 1791680822.42794 |
| Computers | locked | 28831.846 | 1791680836.9338088 |
| Cricket_X | locked | 28846.532 | 1791680851.8256822 |
| Cricket_Y | locked | 28861.270 | 1791680866.7355592 |
| Cricket_Z | locked | 28875.743 | 1791680881.3715215 |
| DistalPhalanxOutlineAgeGroup | locked | 28890.395 | 1791680896.1906135 |
| DistalPhalanxOutlineCorrect | locked | 28904.337 | 1791680910.2949584 |
| DistalPhalanxTW | locked | 28918.920 | 1791680925.0492067 |
| ECG200 | locked | 28933.228 | 1791680939.5219936 |
| ECG5000 | locked | 28947.926 | 1791680954.3855221 |
| Earthquakes | locked | 28962.482 | 1791680969.1042309 |
| ElectricDevices | locked | 28976.812 | 1791680983.603883 |
| FISH | locked | 28991.915 | 1791680998.8703675 |
| FaceAll | locked | 29005.905 | 1791681013.0291164 |
| FaceFour | locked | 29020.153 | 1791681027.4412208 |
| FacesUCR | locked | 29035.176 | 1791681042.6322775 |
| FordA | locked | 29049.867 | 1791681057.4883044 |
| FordB | locked | 29064.055 | 1791681071.8414202 |
| Gun_Point | locked | 29078.208 | 1791681086.164022 |
| Ham | locked | 29093.023 | 1791681101.141457 |
| HandOutlines | locked | 29107.166 | 1791681115.4505942 |
| Haptics | locked | 29122.213 | 1791681130.6677496 |
| Herring | locked | 29136.881 | 1791681145.498638 |
| InlineSkate | locked | 29150.701 | 1791681159.4844408 |
| InsectWingbeatSound | locked | 29165.485 | 1791681174.4546018 |
| ItalyPowerDemand | locked | 29180.004 | 1791681189.1583352 |
| LargeKitchenAppliances | locked | 29194.875 | 1791681204.1940029 |
| Lighting2 | locked | 29209.294 | 1791681218.7837105 |
| Lighting7 | locked | 29223.474 | 1791681233.1261358 |
| MALLAT | locked | 29238.686 | 1791681248.505717 |
| MedicalImages | locked | 29252.690 | 1791681262.6784132 |
| MiddlePhalanxOutlineAgeGroup | locked | 29267.495 | 1791681277.6466622 |
| MiddlePhalanxOutlineCorrect | locked | 29282.161 | 1791681292.4765801 |
| MiddlePhalanxTW | locked | 29296.806 | 1791681307.2812123 |
| MoteStrain | locked | 29310.638 | 1791681321.274215 |
| NonInvasiveFatalECG_Thorax1 | locked | 29325.242 | 1791681336.0416725 |
| NonInvasiveFatalECG_Thorax2 | locked | 29340.262 | 1791681351.229297 |
| OSULeaf | locked | 29355.022 | 1791681366.154314 |
| PhalangesOutlinesCorrect | locked | 29369.129 | 1791681380.423895 |
| Plane | locked | 29383.638 | 1791681395.0950227 |
| ProximalPhalanxOutlineAgeGroup | locked | 29397.888 | 1791681409.5044506 |
| ProximalPhalanxOutlineCorrect | locked | 29412.802 | 1791681424.5756798 |
| ProximalPhalanxTW | locked | 29427.073 | 1791681439.0133169 |
| RefrigerationDevices | locked | 29441.647 | 1791681453.7664878 |
| ScreenType | locked | 29456.620 | 1791681468.9085248 |
| ShapeletSim | locked | 29470.614 | 1791681483.0661314 |
| ShapesAll | locked | 29485.047 | 1791681497.6706414 |
| SmallKitchenAppliances | locked | 29499.763 | 1791681512.548588 |
| StarLightCurves | locked | 29514.104 | 1791681527.054402 |
| Strawberry | locked | 29528.513 | 1791681541.624684 |
| SwedishLeaf | locked | 29543.061 | 1791681556.3365695 |
| Symbols | locked | 29557.725 | 1791681571.1628795 |
| ToeSegmentation1 | locked | 29572.247 | 1791681585.8610914 |
| ToeSegmentation2 | locked | 29586.805 | 1791681600.6086352 |
| Trace | locked | 29601.232 | 1791681615.200106 |
| TwoLeadECG | locked | 29615.859 | 1791681629.989617 |
| Two_Patterns | locked | 29631.013 | 1791681645.3100097 |
| UWaveGestureLibraryAll | locked | 29645.140 | 1791681659.6057868 |
| WordsSynonyms | locked | 29659.721 | 1791681674.3487854 |
| Worms | locked | 29673.997 | 1791681688.7914412 |
| WormsTwoClass | locked | 29688.410 | 1791681703.3744226 |
| synthetic_control | locked | 29702.904 | 1791681718.0359447 |
| uWaveGestureLibrary_X | locked | 29717.941 | 1791681733.2395093 |
| uWaveGestureLibrary_Y | locked | 29731.885 | 1791681747.3456454 |
| uWaveGestureLibrary_Z | locked | 29746.775 | 1791681762.4125898 |
| wafer | locked | 29761.621 | 1791681777.4322357 |
| yoga | locked | 29775.655 | 1791681791.6298878 |
