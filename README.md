# Fine-tuning DETR on Cityscapes Dataset
- This repository contains the code for fine-tuning the DETR model on the Cityscapes dataset for the object detection task.

![DETR](/resources/detr.jpeg)

### Prepare Dataset
- Take a look at [this repository](https://github.com/SuciuCaesar12/cityscapes-to-coco-format) and follow the steps described in the `README.md` file

### Model details
- Used [`Hugging Face`](https://huggingface.co/docs/transformers/main/en/model_doc/detr) to import the DETR model, 
starting from the pre-trained model on COCO dataset with a ResNet-50 backbone: https://huggingface.co/facebook/detr-resnet-50

### Training details
- Replaced the classification head with a new one and fine-tuned it for a single epoch while keeping the rest of the model frozen to prevent feature corruption. Then, we unfroze the entire model and fine-tuned it for the remaining epochs
- Mainly used `Pytorch Lightning` for training and `Weights & Biases`for tracking the training process and logging the metrics as well as the model checkpoints.
- Trained on 2 Tesla V100 GPUs for 25 epochs with a batch size of 4, running in a distributed setup on a docker container.
- More information can be seen in this [run](https://wandb.ai/suciucezar07/detr/runs/in8k822y?nw=nwusersuciucezar07) on `Weights & Biases`.

