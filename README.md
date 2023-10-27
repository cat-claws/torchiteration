# Torch Iteration - A Lightweight PyTorch Training Toolkit

[![GitHub license](https://img.shields.io/github/license/cat-claws/torchiteration.svg)](https://github.com/cat-claws/torchiteration/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/torchiteration.svg)](https://pypi.org/project/torchiteration/)

Torch Iteration is a versatile PyTorch training toolkit designed to simplify your deep learning projects. With real-time statistics and flexible configuration, it empowers you to train models efficiently and effectively.


## Installation
To install Torch Iteration from PyPI, simply run the following command:

```bash
pip install torchiteration
```

To install Torch Iteration from the source code, use the following command:
```
pip install git+https://github.com/cat-claws/torchiteration/
```

## Table of Contents
- [Installation](#installation)
- [How to Use](#how-to-use)
  - [Example: Training a Model on MNIST](#example-training-a-model-on-mnist)
  - [Visualizing Training Progress](#visualizing-training-progress)
  - [Extending Torch Iteration](#extending-torch-iteration)

## How to use

### Example: Training a Model on MNIST
Get started with Torch Iteration by training a model on the [MNIST](https://github.com/pytorch/examples/blob/main/mnist/main.py) dataset. Below is an example script:

```python
import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step



config = {
	'dataset':'mnist',
	'training_step':'classification_step',
	'batch_size':32,
	'optimizer':'Adadelta',
	'optimizer_config':{
	},
	'scheduler':'StepLR',
	'scheduler_config':{
		'step_size':20,
		'gamma':0.1
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'classification_step',
}

model = torch.hub.load('cat-claws/nn', 'exampleconvnet', in_channels = 1).to(config['device'])

writer = SummaryWriter(comment = f"_{config['dataset']}_{model._get_name()}_{config['training_step']}", flush_secs=10)

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = eval(v)
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

import torchvision

train_set = torchvision.datasets.MNIST('', train=True, download=True, transform=torchvision.transforms.ToTensor())
val_set = torchvision.datasets.MNIST('', train=False, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, num_workers = 4, batch_size = config['batch_size'])
val_loader = torch.utils.data.DataLoader(val_set, num_workers = 4, batch_size = config['batch_size'])


for epoch in range(10):
	if epoch > 0:
		train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	torch.save(model.state_dict(), writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
```

### Visualizing Training Progress
To visualize the training progress, run the following command in your terminal:

```properties
tensorboard --logdir=runs
```
### Extending Torch Iteration
You can extend Torch Iteration by creating your own custom ```*_step``` function in the same input-output format as those in ```steps.py```. Make sure your function ends with ```_step``` for easier integration. Here's a template:

```python
# For now, only one model is accepted, but each step can be very versatile
# net must inherit nn.Module
def customised_step(net, batch, batch_idx, **kw):

	# Each dataloader may be defined differently, so you can handle it case by case
	_your_data_at_each_batch  = batch

	# Process your training, testing, etc.
	_your_loss_perhaps, _whatever_you_want_to_monitor = _your_process(net, _your_data_at_each_batch, **kw)

	return {
		'first output':_your_loss_perhaps,
		'second output':_whatever_you_want_to_monitor,
		'third output, etc.': _just_continue
	}
```
With Torch Iteration, streamline your PyTorch training workflows, and easily customize your training steps for your specific project needs.

