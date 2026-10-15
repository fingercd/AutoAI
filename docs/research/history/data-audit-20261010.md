> 2026-10-15 纳入 Git 的历史材料。保留原报告日期、当时结论和方法代号；不是新的执行授权。现状以仓库 `docs/research/README.md` 为准。旧运行路径所指完整产物仍在服务器保留，未将原始数据或日志纳入 Git。

# 第一轮候选数据审计

候选 149 项；保留 129 项，其中开发 15 项、正式 114 项；排除 20 项。

分区预验收共 315 个测试轮次。保留任务使用完整来源记录，已知物理组整体划分；来源记录划分的限制逐项登记。此文件是数据审计，不代表训练完成或效果结论。

|任务|来源|角色|记录数|划分单位|测试轮数|状态/原因|
|---|---|---|---:|---|---:|---|
|analcatdata_aids|tabmini25|development|50|source_record|5|ready|
|analcatdata_asbestos|tabmini25|locked|83|source_record|5|ready|
|analcatdata_bankruptcy|tabmini25|locked|50|source_record|5|ready|
|analcatdata_creditscore|tabmini25|locked|100|source_record|5|ready|
|analcatdata_fraud|tabmini25|locked|42|source_record|5|ready|
|analcatdata_japansolvent|tabmini25|locked|52|source_record|5|ready|
|labor|tabmini25|locked|57|source_record|5|ready|
|lupus|tabmini25|locked|87|source_record|5|ready|
|parity5|tabmini25|development|32|source_physical_group|5|ready|
|postoperative_patient_data|tabmini25|locked|88|source_record|5|ready|
|appendicitis|tabmini25|development|106|source_physical_group|5|ready|
|backache|tabmini25|locked|180|source_record|5|ready|
|corral|tabmini25|locked|160|source_record|5|ready|
|glass2|tabmini25|development|163|source_physical_group|5|ready|
|hepatitis|tabmini25|locked|155|source_record|5|ready|
|molecular_biology_promoters|tabmini25|locked|106|source_record|5|ready|
|mux6|tabmini25|locked|128|source_physical_group|5|ready|
|prnn_crabs|tabmini25|development|200|source_physical_group|5|ready|
|biomed|tabmini25|locked|209|source_record|5|ready|
|breast_cancer|tabmini25|locked|286|source_record|5|ready|
|prnn_synth|tabmini25|locked|250|source_record|5|ready|
|bupa|tabmini25|locked|345|source_physical_group|5|ready|
|haberman|tabmini25|locked|306|source_physical_group|5|ready|
|irish|tabmini25|locked|500|source_record|5|ready|
|saheart|tabmini25|locked|462|source_record|5|ready|
|50words|ucr85|excluded|905|source_record|—|官方Train有单例类别，不能同时覆盖Train与Valid: {'41': 1}|
|Adiac|ucr85|locked|781|source_record|1|ready|
|ArrowHead|ucr85|locked|211|source_record|1|ready|
|Beef|ucr85|development|60|source_record|1|ready|
|BeetleFly|ucr85|development|40|source_physical_group|1|ready|
|BirdChicken|ucr85|development|40|source_physical_group|1|ready|
|CBF|ucr85|locked|930|source_physical_group|1|ready|
|Car|ucr85|locked|120|source_record|1|ready|
|ChlorineConcentration|ucr85|locked|4307|source_record|1|ready|
|CinC_ECG_torso|ucr85|locked|1420|source_record|1|ready|
|Coffee|ucr85|development|56|source_record|1|ready|
|Computers|ucr85|locked|500|source_record|1|ready|
|Cricket_X|ucr85|locked|780|source_record|1|ready|
|Cricket_Y|ucr85|locked|780|source_record|1|ready|
|Cricket_Z|ucr85|locked|780|source_record|1|ready|
|DiatomSizeReduction|ucr85|excluded|322|source_record|—|官方Train有单例类别，不能同时覆盖Train与Valid: {'1': 1}|
|DistalPhalanxOutlineAgeGroup|ucr85|locked|539|source_record|1|ready|
|DistalPhalanxOutlineCorrect|ucr85|locked|876|source_record|1|ready|
|DistalPhalanxTW|ucr85|locked|539|source_record|1|ready|
|ECG200|ucr85|locked|200|source_record|1|ready|
|ECG5000|ucr85|locked|5000|source_record|1|ready|
|ECGFiveDays|ucr85|development|884|source_record|1|ready|
|Earthquakes|ucr85|locked|461|source_physical_group|1|ready|
|ElectricDevices|ucr85|locked|16637|source_record|1|ready|
|FISH|ucr85|locked|350|source_record|1|ready|
|FaceAll|ucr85|locked|2250|source_record|1|ready|
|FaceFour|ucr85|locked|112|source_record|1|ready|
|FacesUCR|ucr85|locked|2250|source_record|1|ready|
|FordA|ucr85|locked|4921|source_record|1|ready|
|FordB|ucr85|locked|4446|source_record|1|ready|
|Gun_Point|ucr85|locked|200|source_record|1|ready|
|Ham|ucr85|locked|214|source_record|1|ready|
|HandOutlines|ucr85|locked|1370|source_record|1|ready|
|Haptics|ucr85|locked|463|source_record|1|ready|
|Herring|ucr85|locked|128|source_record|1|ready|
|InlineSkate|ucr85|locked|650|source_record|1|ready|
|InsectWingbeatSound|ucr85|locked|2200|source_record|1|ready|
|ItalyPowerDemand|ucr85|locked|1096|source_record|1|ready|
|LargeKitchenAppliances|ucr85|locked|750|source_record|1|ready|
|Lighting2|ucr85|locked|121|source_record|1|ready|
|Lighting7|ucr85|locked|143|source_record|1|ready|
|MALLAT|ucr85|locked|2400|source_record|1|ready|
|Meat|ucr85|development|120|source_physical_group|1|ready|
|MedicalImages|ucr85|locked|1141|source_record|1|ready|
|MiddlePhalanxOutlineAgeGroup|ucr85|locked|554|source_record|1|ready|
|MiddlePhalanxOutlineCorrect|ucr85|locked|891|source_record|1|ready|
|MiddlePhalanxTW|ucr85|locked|553|source_record|1|ready|
|MoteStrain|ucr85|locked|1272|source_record|1|ready|
|NonInvasiveFatalECG_Thorax1|ucr85|locked|3765|source_record|1|ready|
|NonInvasiveFatalECG_Thorax2|ucr85|locked|3765|source_record|1|ready|
|OSULeaf|ucr85|locked|442|source_record|1|ready|
|OliveOil|ucr85|development|60|source_physical_group|1|ready|
|PhalangesOutlinesCorrect|ucr85|locked|2658|source_record|1|ready|
|Phoneme|ucr85|excluded|2110|source_record|—|官方Train有单例类别，不能同时覆盖Train与Valid: {'10': 1, '26': 1, '32': 1, '33': 1, '37': 1, '39': 1, '5': 1, '8': 1}|
|Plane|ucr85|locked|210|source_record|1|ready|
|ProximalPhalanxOutlineAgeGroup|ucr85|locked|605|source_record|1|ready|
|ProximalPhalanxOutlineCorrect|ucr85|locked|891|source_record|1|ready|
|ProximalPhalanxTW|ucr85|locked|605|source_record|1|ready|
|RefrigerationDevices|ucr85|locked|750|source_record|1|ready|
|ScreenType|ucr85|locked|750|source_record|1|ready|
|ShapeletSim|ucr85|locked|200|source_physical_group|1|ready|
|ShapesAll|ucr85|locked|1200|source_record|1|ready|
|SmallKitchenAppliances|ucr85|locked|750|source_record|1|ready|
|SonyAIBORobotSurface|ucr85|development|621|source_record|1|ready|
|SonyAIBORobotSurfaceII|ucr85|development|980|source_record|1|ready|
|StarLightCurves|ucr85|locked|9236|source_record|1|ready|
|Strawberry|ucr85|locked|983|source_record|1|ready|
|SwedishLeaf|ucr85|locked|1125|source_record|1|ready|
|Symbols|ucr85|locked|1020|source_record|1|ready|
|ToeSegmentation1|ucr85|locked|268|source_record|1|ready|
|ToeSegmentation2|ucr85|locked|166|source_record|1|ready|
|Trace|ucr85|locked|200|source_physical_group|1|ready|
|TwoLeadECG|ucr85|locked|1162|source_record|1|ready|
|Two_Patterns|ucr85|locked|5000|source_physical_group|1|ready|
|UWaveGestureLibraryAll|ucr85|locked|4478|source_record|1|ready|
|Wine|ucr85|development|111|source_physical_group|1|ready|
|WordsSynonyms|ucr85|locked|905|source_record|1|ready|
|Worms|ucr85|locked|258|source_record|1|ready|
|WormsTwoClass|ucr85|locked|258|source_record|1|ready|
|synthetic_control|ucr85|locked|600|source_physical_group|1|ready|
|uWaveGestureLibrary_X|ucr85|locked|4478|source_record|1|ready|
|uWaveGestureLibrary_Y|ucr85|locked|4478|source_record|1|ready|
|uWaveGestureLibrary_Z|ucr85|locked|4478|source_record|1|ready|
|wafer|ucr85|locked|7164|source_record|1|ready|
|yoga|ucr85|locked|3300|source_record|1|ready|
|ALLAML|biology5|locked|72|source_physical_group|5|ready|
|colon|biology5|locked|62|source_record|5|ready|
|GLIOMA|biology5|locked|50|source_record|5|ready|
|CLL_SUB_111|biology5|locked|111|source_record|5|ready|
|Carcinom|biology5|locked|174|source_record|5|ready|
|pharmaceutical_ingredients|raman34|locked|3510|source_record|5|ready|
|serum_prostate_cancer|raman34|excluded|12601|source_physical_group|—|独立样品重复测量次数不一致；不能删行凑齐|
|serum_alzheimer_disease|raman34|locked|3417|source_physical_group|3|ready|
|comfile_stroke|raman34|locked|4020|source_physical_group|5|ready|
|covid19_salvia|raman34|excluded|2501|source_physical_group|—|独立样品重复测量次数不一致；不能删行凑齐|
|parkinson|raman34|excluded|1476|source_physical_group|—|独立样品重复测量次数不一致；不能删行凑齐|
|alzheimer|raman34|excluded|1151|source_physical_group|—|独立样品重复测量次数不一致；不能删行凑齐|
|chlorinated_samples|raman34|locked|230|source_record|5|ready|
|biomolecules_reference|raman34|excluded|202|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 1, '101': 1, '102': 1, '103': 1, '104': 1, '105': 2, '106': 1, '108': 1, '11': 2, '110': 1, '111': 1, '112': 1, '113': 1, '114': 1, '115': 1, '116': 1, '11…（完整依据见 manifest.json）|
|rruff_mineral_raw|raman34|excluded|5244|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 2, '100': 2, '1000': 1, '1002': 2, '1005': 1, '1006': 2, '1007': 2, '1008': 2, '101': 1, '1010': 1, '1013': 2, '1014': 1, '1017': 1, '1018': 1, '102': 1, '…（完整依据见 manifest.json）|
|rruff_mineral_preprocess|raman34|excluded|27801|source_record|—|该来源是已处理的光谱匹配参考库/训练验证数组，原始采集与处理后记录的对应关系未保留；作者实现含训练谱增强，当前包不能核验哪些记录属于同一原始测量。无法形成可复核的独立分类划分，本轮不纳入。|
|organic_compounds_raw|raman34|excluded|377|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 1, '10': 1, '100': 1, '101': 1, '102': 1, '103': 1, '104': 1, '105': 1, '106': 1, '107': 1, '108': 1, '109': 1, '11': 1, '110': 1, '111': 1, '112': 1, '113…（完整依据见 manifest.json）|
|organic_compounds_preprocess|raman34|excluded|5278|source_record|—|该来源是已处理的光谱匹配参考库/训练验证数组，原始采集与处理后记录的对应关系未保留；作者实现含训练谱增强，当前包不能核验哪些记录属于同一原始测量。无法形成可复核的独立分类划分，本轮不纳入。|
|diabetes_skin_ages|raman34|excluded|6|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}|
|diabetes_skin_ear_lobe|raman34|locked|20|source_physical_group|5|ready|
|diabetes_skin_inner_arm|raman34|locked|20|source_physical_group|5|ready|
|diabetes_skin_thumbnail|raman34|locked|20|source_physical_group|5|ready|
|diabetes_skin_vein|raman34|locked|20|source_physical_group|5|ready|
|cancer_cell_cooh|raman34|locked|633|source_record|5|ready|
|cancer_cell_nh2|raman34|locked|632|source_record|5|ready|
|cancer_cell_(cooh)2|raman34|locked|627|source_record|5|ready|
|illicit_adulterants_ft_raman|raman34|excluded|11|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 1, '10': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1}|
|illicit_adulterants_sers|raman34|excluded|11|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'0': 1, '1': 1, '10': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1}|
|microplastics_weathered|raman34|excluded|173|source_record|—|完整数据类别组数不足以覆盖Train、Valid、Test: {'1': 1, '2': 1, '4': 2, '8': 2}|
|covid19_serum|raman34|excluded|None|source_record|—|登记来源只允许联系作者取得covid19.xls；本轮没有可用来源资产，也未取得发送邮件的授权。按用户允许不使用无法合理评测任务的要求登记排除，保留来源与排除原因。|
|bacteria_identification|raman34|excluded|None|source_record|—|固定来源加载器直接合并30菌株标签与8治疗分组标签，整数标签含义冲突；本轮不改默认目标、不重映射临床标签，按用户授权剔除。|
|marine_pathogens|raman34|locked|1138|source_record|5|ready|
|marine_pathogens_binary|raman34|locked|261|source_record|5|ready|
|mlrod|raman34|excluded|130061|未应用|—|原加载器得到130061条记录、16类，单类至少4817条；属于大规模光谱分类矩阵，与本轮小样本模型选择主实验范围不匹配。保留完整来源资产，本轮不通过抽样缩小后纳入。|
|wheat_lines|raman34|excluded|53134|未应用|—|原加载器得到53134条记录、4类，单类至少12659条；属于大规模光谱分类矩阵，与本轮小样本模型选择主实验范围不匹配。保留完整来源资产，本轮不通过抽样缩小后纳入。|
|hair_dyes_sers|raman34|locked|1713|source_record|5|ready|
|head_neck_cancer|raman34|locked|111|source_record|5|ready|
|locust_phase_hemolymph|raman34|locked|5000|source_physical_group|5|ready|
|cspp_serum_metabolites|raman34|locked|150|source_physical_group|5|ready|
