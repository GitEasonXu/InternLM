# 第六节：OpenCompass 大模型评测



## 目录

- 环境配置
- 模型评测



### 1. 环境配置

```bash
# GPU环境
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```



### 2. 模型评测

- 使用`list_configs.py`查看当前模型和测试集对应的测试配置

  ```bash
  #                             模型名   评测数据
  python tools/list_configs.py internlm ceval
  ```

- 输入以下指令完成评测

  ```bash
  python run.py \
  --datasets ceval_gen \
  --hf-path /share/temp/model_repos/internlm-chat-7b/ \   
  --tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \
  --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
  --model-kwargs trust_remote_code=True device_map='auto' \
  --max-seq-len 2048 \
  --max-out-len 16 \
  --batch-size 4 \
  --num-gpus 1 \  
  --debug
  ```

  评测结束后会将结果保存到本地，并且终端也会输出如下内容:

  ```bash
  dataset                                         version    metric         mode      opencompass.models.huggingface.HuggingFace_model_repos_internlm-chat-7b
  ----------------------------------------------  ---------  -------------  ------  -------------------------------------------------------------------------
  ceval-computer_network                          db9ce2     accuracy       gen                                                                         36.84
  ceval-operating_system                          1c2571     accuracy       gen                                                                         36.84
  ceval-computer_architecture                     a74dad     accuracy       gen                                                                         28.57
  ceval-college_programming                       4ca32a     accuracy       gen                                                                         32.43
  ceval-college_physics                           963fa8     accuracy       gen                                                                         31.58
  ceval-college_chemistry                         e78857     accuracy       gen                                                                         16.67
  ceval-advanced_mathematics                      ce03e2     accuracy       gen                                                                         21.05
  ceval-probability_and_statistics                65e812     accuracy       gen                                                                         38.89
  ceval-discrete_mathematics                      e894ae     accuracy       gen                                                                         18.75
  ceval-electrical_engineer                       ae42b9     accuracy       gen                                                                         35.14
  ceval-metrology_engineer                        ee34ea     accuracy       gen                                                                         50
  ceval-high_school_mathematics                   1dc5bf     accuracy       gen                                                                         22.22
  ceval-high_school_physics                       adf25f     accuracy       gen                                                                         31.58
  ceval-high_school_chemistry                     2ed27f     accuracy       gen                                                                         15.79
  ceval-high_school_biology                       8e2b9a     accuracy       gen                                                                         36.84
  ceval-middle_school_mathematics                 bee8d5     accuracy       gen                                                                         26.32
  ceval-middle_school_biology                     86817c     accuracy       gen                                                                         61.9
  ceval-middle_school_physics                     8accf6     accuracy       gen                                                                         63.16
  ceval-middle_school_chemistry                   167a15     accuracy       gen                                                                         65
  ceval-veterinary_medicine                       b4e08d     accuracy       gen                                                                         47.83
  ceval-college_economics                         f3f4e6     accuracy       gen                                                                         38.18
  ceval-business_administration                   c1614e     accuracy       gen                                                                         33.33
  ceval-marxism                                   cf874c     accuracy       gen                                                                         68.42
  ceval-mao_zedong_thought                        51c7a4     accuracy       gen                                                                         70.83
  ceval-education_science                         591fee     accuracy       gen                                                                         58.62
  ceval-teacher_qualification                     4e4ced     accuracy       gen                                                                         68.18
  ceval-high_school_politics                      5c0de2     accuracy       gen                                                                         26.32
  ceval-high_school_geography                     865461     accuracy       gen                                                                         47.37
  ceval-middle_school_politics                    5be3e7     accuracy       gen                                                                         52.38
  ceval-middle_school_geography                   8a63be     accuracy       gen                                                                         58.33
  ceval-modern_chinese_history                    fc01af     accuracy       gen                                                                         73.91
  ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen                                                                         63.16
  ceval-logic                                     f5b022     accuracy       gen                                                                         31.82
  ceval-law                                       a110a1     accuracy       gen                                                                         25
  ceval-chinese_language_and_literature           0f8b68     accuracy       gen                                                                         30.43
  ceval-art_studies                               2a1300     accuracy       gen                                                                         60.61
  ceval-professional_tour_guide                   4e673e     accuracy       gen                                                                         62.07
  ceval-legal_professional                        ce8787     accuracy       gen                                                                         39.13
  ceval-high_school_chinese                       315705     accuracy       gen                                                                         57.89
  ceval-high_school_history                       7eb30a     accuracy       gen                                                                         70
  ceval-middle_school_history                     48ab4a     accuracy       gen                                                                         59.09
  ceval-civil_servant                             87d061     accuracy       gen                                                                         53.19
  ceval-sports_science                            70f27b     accuracy       gen                                                                         52.63
  ceval-plant_protection                          8941f9     accuracy       gen                                                                         59.09
  ceval-basic_medicine                            c409d6     accuracy       gen                                                                         47.37
  ceval-clinical_medicine                         49e82d     accuracy       gen                                                                         40.91
  ceval-urban_and_rural_planner                   95b885     accuracy       gen                                                                         45.65
  ceval-accountant                                002837     accuracy       gen                                                                         26.53
  ceval-fire_engineer                             bc23f5     accuracy       gen                                                                         22.58
  ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen                                                                         64.52
  ceval-tax_accountant                            3a5e3c     accuracy       gen                                                                         34.69
  ceval-physician                                 6e277d     accuracy       gen                                                                         44.9
  ceval-stem                                      -          naive_average  gen                                                                         35.87
  ceval-social-science                            -          naive_average  gen                                                                         52.2
  ceval-humanities                                -          naive_average  gen                                                                         52.1
  ceval-other                                     -          naive_average  gen                                                                         44.73
  ceval-hard                                      -          naive_average  gen                                                                         24.57
  ceval                                           -          naive_average  gen                                                                         44.32
  01/20 18:10:57 - OpenCompass - INFO - write summary to /root/code/InternLM/homework/course_6/opencompass-0.2.1/outputs/default/20240120_173228/summary/summary_20240120_173228.txt
  01/20 18:10:57 - OpenCompass - INFO - write csv to /root/code/InternLM/homework/course_6/opencompass-0.2.1/outputs/default/20240120_173228/summary/summary_20240120_173228.csv
  ```