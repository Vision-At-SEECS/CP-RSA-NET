
<!-- ![Qualitative Results](https://user-images.githubusercontent.com/86110742/184842418-0f2906e2-6a13-4c58-9d5c-9f57267f07d0.PNG) -->

# Recursive Skip Attention Network with mask refinement for nuclei instance Segmentation and classification in histology images
Automatic nuclei instance segmentation and classification within Hematoxylin \& Eosin stained images can be considered a preliminary step in developing advanced medical systems which can provide diagnosis and prognosis to deadly diseases like cancer. However, the heterogeneous and crowding (overlapping and touching nuclei) nature of the cells (with large inter and intra class variability), combined with the fact that there are currently no reproducible measures to evaluate a patient's biopsy, makes the digital profiling of tumor micro environments even more challenging. To address these challenges, we have proposed a novel deep multi-branch CNN for simultaneous segmentation and classification of nuclei in H\&E stained histopathology images. The network is composed of a shared encoder and multi-branch decoder architecture with embedded Recursive Skip Attention (RSA) blocks and a novel mask refinement step. Alongside the segmentation and classification mask, the network also learns the horizontal and vertical pixel distances for each of the nuclei instances from their center of masses to isolate clustered and overlapping nuclei instances. The RSA blocks hooked in from the residual blocks to the decoder blocks makes the network focus more on the significant features, and that too at a very low parameter cost. The model exhibits competitive performance against SOTA methods that too on various different publicly available histology H\&E stained image datasets. Additionally, we have also performed an additional mask refinement step along with the post-processing to make the network predictions even more certain.

# Model Architecture

![RSA-Net](https://user-images.githubusercontent.com/86110742/184842418-0f2906e2-6a13-4c58-9d5c-9f57267f07d0.PNG)

# Trained Weights
The trained weights can be downloaded from [here](https://drive.google.com/file/d/1uVjBAQxOyOC3w96w95TPO06LQJjCKvWI/view?usp=sharing)

# Running the Code
## To Set Up Environment

```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```
or install via the ``` requirements.txt ``` file

## Inference
Run the following command in the terminal:
```
python run_infer.py --batch_size='16' --model_path={Ckpt_path} --model_mode={model_mode} tile --input_dir={test_path}  --output_dir={save_path}  --draw_dot --save_raw_map 
```
where,
``` Ckpt_path : your path to the trained model file.
    model_mode: 'fast' for instance segmentation classification, 'original' for instance segmentation only.
    test_path: Path to the directory containing your test images.
    output_dir: Path to the output directory, where you want the result to be saved.
```

## Training
Prerequisites:
- Set path to training data in `config.py`
- Set checkpoints path in `config.py`
- Set pretrained network weights in `models/hovernet/opt.py`. 
- Hyperparameters can be modified in `models/hovernet/opt.py`.

Run the following command in the terminal:
```
python run_train.py --gpu='0'
```
Note: you can provide as many gpus by referring to their IDs in the --gpu flag for parallel processing. for example: --gpu='0,1,2'
