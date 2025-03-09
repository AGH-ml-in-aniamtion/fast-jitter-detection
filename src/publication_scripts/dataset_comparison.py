from analysis.dataset_analysis import calculate_metrics
from analysis.dataset_analyzer import DatasetConfigurator

if __name__ == "__main__":
    ds_configs = [
        "data/datasets/publication_configs/dataset_contemporary.json",
        "data/datasets/publication_configs/dataset_lafan1.json",
        "data/datasets/publication_configs/dataset_bandai-1.json",
        "data/datasets/publication_configs/dataset_bandai-2.json",
        "data/datasets/publication_configs/dataset_h36m.json",
    ]

    for ds_conf in ds_configs:
        dataset_analyzer = DatasetConfigurator(dataset_config_path=ds_conf)
        calculate_metrics(dataset_analyzer, noise_windows=None, analyze_errors_and_warnings=True)


# Contemporary Dance:
#
# MDC metric (averaged over joints and then over frames): 0.40757808089256287
# MDCSS metric (averaged over joints and then over frames): 8.108827590942383
# Error intervals: 7543
# Error frames: 85372
# Warning intervals: 17563
# Warning frames: 77892
#
#
# LAFAN1:
#
# MDC metric (averaged over joints and then over frames): 0.11288538575172424
# MDCSS metric (averaged over joints and then over frames): 0.46242859959602356
# Error intervals: 6
# Error frames: 16
# Warning intervals: 140
# Warning frames: 206
# ERROR: Errors on sequence: obstacles5_subject3.bvh (16 errors)
#
#
# Bandai 1
#
# MDC metric (averaged over joints and then over frames): 0.1057245135307312
# MDCSS metric (averaged over joints and then over frames): 0.43145036697387695
# Error intervals: 0
# Error frames: 0
# Warning intervals: 43
# Warning frames: 45
#
# Bandai 2
#
# MDC metric (averaged over joints and then over frames): 0.08389224857091904
# MDCSS metric (averaged over joints and then over frames): 0.40895965695381165
# Error intervals: 0
# Error frames: 0
# Warning intervals: 458
# Warning frames: 469
#
#
# Human3.6M
#
# MDC metric (averaged over joints and then over frames): 0.09860801696777344
# MDCSS metric (averaged over joints and then over frames): 0.3955531716346741
# Error intervals: 17
# Error frames: 21
# Warning intervals: 205
# Warning frames: 355



# ERRORS:
#
# Contemporary Dance:
#
# Errors on sequence: Andria_Angry_v1-01.bvh (254 errors)
# Errors on sequence: Andria_Angry_v2-01.bvh (4 errors)
# Errors on sequence: Andria_Excited_v1-01.bvh (86 errors)
# Errors on sequence: Andria_Excited_v2-01.bvh (197 errors)
# Errors on sequence: Andria_Happy_v1-01.bvh (64 errors)
# Errors on sequence: Andria_Happy_v2-01.bvh (189 errors)
# Errors on sequence: Andria_Miserable_v1-01.bvh (5 errors)
# Errors on sequence: Andria_Miserable_v2-01.bvh (577 errors)
# Errors on sequence: Andria_Pleased_v1-01.bvh (72 errors)
# Errors on sequence: Andria_Pleased_v2-01.bvh (71 errors)
# Errors on sequence: Andria_Relaxed_v1_2-01.bvh (74 errors)
# Errors on sequence: Andria_Relaxed_v2-01.bvh (85 errors)
# Errors on sequence: Andria_Satisfied_v1-01.bvh (30 errors)
# Errors on sequence: Andria_Satisfied_v2-01.bvh (11 errors)
# Errors on sequence: Andria_Tired_v1-01.bvh (329 errors)
# Errors on sequence: Andria_Tired_v2-01.bvh (47 errors)
# Errors on sequence: Anna_Charalambous_Active.bvh (172 errors)
# Errors on sequence: Anna_Charalambous_Angry.bvh (16 errors)
# Errors on sequence: Anna_Charalambous_Curiosity.bvh (42 errors)
# Errors on sequence: Anna_Charalambous_Happy.bvh (922 errors)
# Errors on sequence: Anna_Charalambous_Nervous.bvh (204 errors)
# Errors on sequence: Anna_Charalambous_Sad.bvh (12 errors)
# Errors on sequence: Anna_Charalambous_Scary.bvh (416 errors)
# Errors on sequence: Ballet_Ioanna_Tacha_Believer.bvh (123 errors)
# Errors on sequence: Ballet_Ioanna_Tacha_Piano.bvh (130 errors)
# Errors on sequence: Elena_Afraid_v1-01.bvh (730 errors)
# Errors on sequence: Elena_Angry_v1-01.bvh (856 errors)
# Errors on sequence: Elena_Angry_v2-01.bvh (604 errors)
# Errors on sequence: Elena_Annoyed_v1-01.bvh (497 errors)
# Errors on sequence: Elena_Annoyed_v2-01.bvh (98 errors)
# Errors on sequence: Elena_Bored_v1-01.bvh (46 errors)
# Errors on sequence: Elena_Bored_v2-01.bvh (751 errors)
# Errors on sequence: Elena_Excited_v1_1-01.bvh (44 errors)
# Errors on sequence: Elena_Excited_v2-01.bvh (884 errors)
# Errors on sequence: Elena_Happy_v1-01.bvh (579 errors)
# Errors on sequence: Elena_Happy_v2-01.bvh (465 errors)
# Errors on sequence: Elena_Miserable_v1-01.bvh (92 errors)
# Errors on sequence: Elena_Miserable_v2-01.bvh (480 errors)
# Errors on sequence: Elena_Neutral_v1-01.bvh (171 errors)
# Errors on sequence: Elena_Neutral_v2-01.bvh (48 errors)
# Errors on sequence: Elena_Pleased_v1-01.bvh (121 errors)
# Errors on sequence: Elena_Pleased_v2-01.bvh (168 errors)
# Errors on sequence: Elena_Relaxed_v1-01.bvh (13 errors)
# Errors on sequence: Elena_Sad_v1-01.bvh (16 errors)
# Errors on sequence: Elena_Sad_v2-01.bvh (181 errors)
# Errors on sequence: Elena_Satisfied_v2_2-01.bvh (350 errors)
# Errors on sequence: Elena_Tired_v1-01.bvh (238 errors)
# Errors on sequence: Elena_Tired_v2-01.bvh (112 errors)
# Errors on sequence: Maritsa_Elia_Afraid_2-01.bvh (277 errors)
# Errors on sequence: Maritsa_Elia_Angry_3-01.bvh (214 errors)
# Errors on sequence: Maritsa_Elia_Excited_v1.bvh (1232 errors)
# Errors on sequence: Maritsa_Elia_Happy_v1.bvh (1520 errors)
# Errors on sequence: Maritsa_Elia_Mix-01.bvh (410 errors)
# Errors on sequence: Maritsa_Elia_Relaxed_2-01.bvh (64 errors)
# Errors on sequence: Maritsa_Elia_Sad-01.bvh (354 errors)
# Errors on sequence: Olivia_Kyriakides_Afraid-01.bvh (541 errors)
# Errors on sequence: Olivia_Kyriakides_Angry-01.bvh (1110 errors)
# Errors on sequence: Olivia_Kyriakides_Annoyed_2-01.bvh (622 errors)
# Errors on sequence: Olivia_Kyriakides_Bored-01.bvh (43 errors)
# Errors on sequence: Olivia_Kyriakides_Excited_v01.bvh (3021 errors)
# Errors on sequence: Olivia_Kyriakides_Happy_v01.bvh (1117 errors)
# Errors on sequence: Olivia_Kyriakides_Miserable-01.bvh (840 errors)
# Errors on sequence: Olivia_Kyriakides_Pleased-01.bvh (3543 errors)
# Errors on sequence: Olivia_Kyriakides_Relaxed-01.bvh (188 errors)
# Errors on sequence: Olivia_Kyriakides_Sad_2-01.bvh (1275 errors)
# Errors on sequence: Olivia_Kyriakides_Satisfied_v01.bvh (754 errors)
# Errors on sequence: Olivia_Kyriakides_Tired-01.bvh (405 errors)
# Errors on sequence: Sophie_Afraid-01.bvh (1537 errors)
# Errors on sequence: Sophie_Angry-01.bvh (632 errors)
# Errors on sequence: Sophie_Annoyed-01.bvh (2264 errors)
# Errors on sequence: Sophie_Bored-01.bvh (1129 errors)
# Errors on sequence: Sophie_Excited-01.bvh (1839 errors)
# Errors on sequence: Sophie_Happy-01.bvh (3921 errors)
# Errors on sequence: Sophie_Miserable-01.bvh (1383 errors)
# Errors on sequence: Sophie_Mix-01.bvh (590 errors)
# Errors on sequence: Sophie_Pleased-01.bvh (323 errors)
# Errors on sequence: Sophie_Relaxed_new-01.bvh (1115 errors)
# Errors on sequence: Sophie_Sad-01.bvh (191 errors)
# Errors on sequence: Sophie_Satisfied-01.bvh (6232 errors)
# Errors on sequence: Sophie_Tired-01.bvh (310 errors)
# Errors on sequence: Theodora_Tsiakka_Afraid-01.bvh (2808 errors)
# Errors on sequence: Theodora_Tsiakka_Angry_3-01.bvh (1032 errors)
# Errors on sequence: Theodora_Tsiakka_Annoyed-01.bvh (689 errors)
# Errors on sequence: Theodora_Tsiakka_Bored_3-01.bvh (300 errors)
# Errors on sequence: Theodora_Tsiakka_Excited_v1.bvh (2075 errors)
# Errors on sequence: Theodora_Tsiakka_Happy_v1.bvh (194 errors)
# Errors on sequence: Theodora_Tsiakka_Miserable-01.bvh (3174 errors)
# Errors on sequence: Theodora_Tsiakka_Mix-01.bvh (6519 errors)
# Errors on sequence: Theodora_Tsiakka_Pleased-01.bvh (2539 errors)
# Errors on sequence: Theodora_Tsiakka_Relaxed-01.bvh (1718 errors)
# Errors on sequence: Theodora_Tsiakka_Sad_2-01.bvh (1577 errors)
# Errors on sequence: Theodora_Tsiakka_Satisfied_v1.bvh (353 errors)
# Errors on sequence: Theodora_Tsiakka_Tired-01.bvh (1570 errors)
# Errors on sequence: Vasso_Aristeidou_Afraid-01.bvh (112 errors)
# Errors on sequence: Vasso_Aristeidou_Afraid_v1-01.bvh (316 errors)
# Errors on sequence: Vasso_Aristeidou_Angry-01.bvh (1395 errors)
# Errors on sequence: Vasso_Aristeidou_Angry_v1-01.bvh (183 errors)
# Errors on sequence: Vasso_Aristeidou_Annoyed-01.bvh (116 errors)
# Errors on sequence: Vasso_Aristeidou_Annoyed_v1-01.bvh (42 errors)
# Errors on sequence: Vasso_Aristeidou_Bored-01.bvh (882 errors)
# Errors on sequence: Vasso_Aristeidou_Bored_v1-01.bvh (483 errors)
# Errors on sequence: Vasso_Aristeidou_Excited-01.bvh (281 errors)
# Errors on sequence: Vasso_Aristeidou_Excited_v1-01.bvh (553 errors)
# Errors on sequence: Vasso_Aristeidou_Happy-01.bvh (71 errors)
# Errors on sequence: Vasso_Aristeidou_Happy_v1-01.bvh (37 errors)
# Errors on sequence: Vasso_Aristeidou_Miserable-01.bvh (134 errors)
# Errors on sequence: Vasso_Aristeidou_Miserable_v1-01.bvh (165 errors)
# Errors on sequence: Vasso_Aristeidou_Mix-01.bvh (830 errors)
# Errors on sequence: Vasso_Aristeidou_Neutral_v2-01.bvh (89 errors)
# Errors on sequence: Vasso_Aristeidou_Neutral_v3-01.bvh (270 errors)
# Errors on sequence: Vasso_Aristeidou_Neutral_v4-01.bvh (68 errors)
# Errors on sequence: Vasso_Aristeidou_Pleased-01.bvh (119 errors)
# Errors on sequence: Vasso_Aristeidou_Pleased_v1-01.bvh (519 errors)
# Errors on sequence: Vasso_Aristeidou_Relaxed-01.bvh (8 errors)
# Errors on sequence: Vasso_Aristeidou_Relaxed_v1-01.bvh (266 errors)
# Errors on sequence: Vasso_Aristeidou_Sad-01.bvh (144 errors)
# Errors on sequence: Vasso_Aristeidou_Sad_v1-01.bvh (1800 errors)
# Errors on sequence: Vasso_Aristeidou_Satisfied-01.bvh (258 errors)
# Errors on sequence: Vasso_Aristeidou_Satisfied_v1-01.bvh (792 errors)
# Errors on sequence: Vasso_Aristeidou_Tired_v1-01.bvh (219 errors)
#
#
# Human3.6M
#
# Errors on sequence: Greeting1_subject17.bvh (2 errors)
# Errors on sequence: Greeting1_subject19.bvh (2 errors)
# Errors on sequence: Photo1_subject16.bvh (2 errors)
# Errors on sequence: SittingDown1_subject17.bvh (2 errors)
# Errors on sequence: SittingDown_subject15.bvh (2 errors)
# Errors on sequence: SittingDown_subject16.bvh (3 errors)
# Errors on sequence: Sitting_subject15.bvh (5 errors)
# Errors on sequence: WalkDog1_subject19.bvh (3 errors)
