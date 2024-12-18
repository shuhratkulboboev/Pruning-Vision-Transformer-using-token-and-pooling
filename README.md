# Pruning-Vision-Transformer-using-token-and-pooling

The proposed framework, named token Pruning & Pooling Transformers (PPT), allows you to take an existing Vision Transformer architecture and efficiently compress tokens inside of the network for faster evaluation. PPT is tuned to seamlessly fit inside existing vision transformers, so you can use it without having to do additional training. And if you do use PPT during training, you can reduce the accuracy drop even further while also speeding up training considerably.

## Important
All explanation regarding this method mentioned inmy thesis work which I attached above. Some results and techniques and bad and good side of this method explained with mathematical formula. This project is still ongoing and I did not get final result of this method and I am trying to optimize it more and more now.
## Usage
1. Clone the repository locally:
   ```bash
   git clone https://github.com/shuhratkulboboev/Pruning-Vision-Transformer-using-token-and-pooling.git

2. Data preparation:
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision datasets.ImageFolder, and the training and validation data is expected to be in the train/ folder and val folder respectively:
   ```bash
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg

3. Evaluation:
To evaluate PPT on a pre-trained DeiT-small (without fine-tuning) on ImageNet val with a single GPU run:
   ```bash
python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --data-path /path/to/imagenet --batch_size 256 --r_tokens 50 --pp_loc_list 3 6 9 --threshold 7e-5

4. Training :
To fine-tuning DeiT-small on ImageNet on a single node with 1 gpus for 50 epochs run:
   ```bash
python -m torch.distributed.launch  --nproc_per_node=1 --use_env main.py --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --batch_size 256 --data-path /path/to/imagenet --epochs 50 --output_dir outputs/PPT_DeiT-S_thr-6e-5_r-50_lr-1e-5 --lr 1e-5 --r_tokens 50 --threshold 6e-5 --pp_loc_list 3 6 9
