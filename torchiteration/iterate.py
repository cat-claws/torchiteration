import torch

def train(net, training_step, train_loader, optimizer, scheduler, **kw):
	net.train()
	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		output = training_step(net, batch, batch_idx, **kw)
		loss = output['loss'] / kw['batch_size']
		loss.backward()
		optimizer.step()
		outputs.append({k:v.detach().cpu() for k, v in output.items()})
		for k, v in output.items():
			kw['writer'].add_scalar("Step-" + k + "-train", v / kw['batch_size'], kw['epoch'] * len(train_loader) + batch_idx)


	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/train", v / len(train_loader.dataset), kw['epoch'])
	
	scheduler.step()

def train_cast_clip(net, training_step, train_loader, optimizer, scheduler, **kw):
	net.train()
	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		with torch.autocast(kw['device']):
			output = training_step(net, batch, batch_idx, **kw)
		loss = output['loss'] / kw['batch_size']
		loss.backward()
		torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
		optimizer.step()
		outputs.append({k:v.detach().cpu() for k, v in output.items()})
		for k, v in output.items():
			kw['writer'].add_scalar("Step-" + k + "-train", v / kw['batch_size'], kw['epoch'] * len(train_loader) + batch_idx)


	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/train", v / len(train_loader.dataset), kw['epoch'])
	
	scheduler.step()

def train_plain(net, training_step, train_loader, optimizer, scheduler, **kw):
	net.train()
	outputs = []
	for batch_idx, batch in enumerate(train_loader):
		optimizer.zero_grad()
		output = training_step(net, batch, batch_idx, **kw)
		optimizer.step()
		outputs.append({k:v.detach().cpu() for k, v in output.items()})
		for k, v in output.items():
			kw['writer'].add_scalar("Step-" + k + "-train", v / kw['batch_size'], kw['epoch'] * len(train_loader) + batch_idx)


	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/train", v / len(train_loader.dataset), kw['epoch'])
	
	scheduler.step()

def validate(net, validation_step, val_loader, **kw):
	net.eval()
	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = validation_step(net, batch, batch_idx, **kw)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})
			for k, v in output.items():
				kw['writer'].add_scalar("Step-" + k + "-valid", v / kw['batch_size'], kw['epoch'] * len(val_loader) + batch_idx)

	outputs = {k: sum([dic[k] for dic in outputs]) for k in outputs[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/valid", v / len(val_loader.dataset), kw['epoch'])


def attack(net, validation_step, attacked_step, val_loader, **kw):
  
	net.eval()
	outputs = []
	outputs_ = []
	for batch_idx, batch in enumerate(val_loader):
		with torch.no_grad():
			output = validation_step(net, batch, batch_idx, **kw)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})
			for k, v in output.items():
				kw['writer'].add_scalar("Step-" + k + "-valid", v / kw['batch_size'], kw['epoch'] * len(val_loader) + batch_idx)

		output_ = attacked_step(net, batch, batch_idx, **kw)
		outputs_.append({k:v.detach().cpu() for k, v in output_.items()})
		for k, v in output.items():
			kw['writer'].add_scalar("Step-" + k + "-attack", v / kw['batch_size'], kw['epoch'] * len(val_loader) + batch_idx)

	outputs = {k: sum([dic[k] for dic in outputs]).item() for k in outputs[0]}
	outputs_ = {k: sum([dic[k] for dic in outputs_]).item() for k in outputs_[0]}
	for k, v in outputs.items():
		kw['writer'].add_scalar("Epoch-" + k + "/valid", v / len(val_loader.dataset), kw['epoch'])
	for k, v in outputs_.items():
		kw['writer'].add_scalar("Epoch-" + k + "/attack", v / len(val_loader.dataset), kw['epoch'])

	return outputs, outputs_


def predict(net, predict_step, val_loader, **kw):
	net.eval()
	outputs = []
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			output = predict_step(net, batch, batch_idx, **kw)
			outputs.append({k:v.detach().cpu() for k, v in output.items()})

	outputs = {k: torch.cat([dic[k] for dic in outputs], dim = 0).tolist() for k in outputs[0]} # array outputs

	return outputs
