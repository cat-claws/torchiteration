# Torch iteration
A lightweight PyTorch training toolkit.

## Installation
To install this small tool from the source code
```
pip install git+https://github.com/cat-claws/torchiteration/
```

## How to use
Example for [MNIST](https://github.com/pytorch/examples/blob/main/mnist/main.py)
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

	else:
		validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	torch.save(model.state_dict(), writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
```

To see the result, run this command in Terminal:
```properties
tensorboard --logdir=runs
```

To extend the usage, you can write your own ```*_step``` function in the same input-output format as that in ```steps.py```. Essentially, the format is below. Note that, you'd better let your function end with ```_step``` to make things easier.
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
