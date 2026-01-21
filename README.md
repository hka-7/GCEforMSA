# GCE 👋
Zijing Huang, Wen-Jue He, Baotian Hu, Zheng Zhang*, Grading-Inspired Complementary Enhancing for Multimodal Sentiment Analysis , Information Fusion, 2026. (Accepted)
<!--
**GCEforMSA/GCEforMSA** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

## Introduction
Inspired by the paradigm of knowledge distillation where a proficient 'teacher' model transfers knowledge to a 'student' model, we extend this idea to cross-modal fusion and propose a guidance mechanism that enhances complementary learning between modality pairs. In standard knowledge distillation, the guidance direction is fixed because the teacher is consistently superior in performance. However, in multimodal sentiment analysis, the discriminative power of each modality or modality pair can vary across samples and domains. Thus, using a predetermined guidance direction is insufficiently robust for diverse scenarios. To address this, we introduce a task-aware grading mechanism that dynamically selects the most discriminative modality pair as the guiding branch based on their actual task performance. Furthermore, unlike conventional feature-level distillation that seeks alignment between teacher and student features, our goal in multimodal fusion is to exploit and preserve complementary information. To this end, we design a relation filtering mechanism that suppresses redundant relations while enhancing informative and complementary interactions during cross-modal learning.
<img width="2433" height="1140" alt="modal-pic1-en17-1" src="https://github.com/user-attachments/assets/28c419bf-4a69-40c6-9664-96d945cd2fba" />



## Usage
### Training
1. Download the CMU-MOSI and CMU-MOSEI dataset from
   - [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK)
   - [Baidu Disk](https://pan.baidu.com/share/init?surl=Wxo4Bim9JhNmg8265p3ttQ) (Extraction code: `g3m2`)
   
   Place them under the folder `/GCEforMSA/datasets` 

2. Download the pre-trained `bert-base-uncased` model from: [huggingface](https://huggingface.co/google-bert/bert-base-uncased)

   Place them under the folder `/GCEforMSA/BERT` 
   

4. Modify the following paths in `/GCEforMSA/src/config.py` to match your local file locations:
   - `MODEL_PATH`: Path to the downloaded `bert-base-uncased` model  (under `/GCEforMSA/BERT`)
   - `MOSI_PATH`: Path to the CMU-MOSI dataset (under `/GCEforMSA/datasets`)
   - `MOSEI_PATH`: Path to the CMU-MOSEI dataset (under `/GCEforMSA/datasets`)

5. Run the following commands in your terminal:
   ```bash
   cd /GCEforMSA/src
   sh run.sh

### Testing
We also provide some pretrained models for testing. ([Google drive](https://drive.google.com/file/d/1l8lBr0TtTUlNvJuAiFE1qti0adp7-cK2/view?usp=sharing))

## Citation
Please cite our paper if you find our work useful for your research.
