import inspect
from torch.utils.tensorboard.summary import hparams

def get_kwargs_for(cls, config, prefix):
    sig = inspect.signature(cls)
    return {
        k[len(prefix)+1:]: v
        for k, v in config.items()
        if k.startswith(prefix + "_") and k[len(prefix)+1:] in sig.parameters
    }

def build_optimizer(config, params):
    cls = getattr(torch.optim, config['optimizer'])
    return cls(params, **get_kwargs_for(cls, config, "optimizer"))

def build_scheduler(config, optimizer):
    cls = getattr(schetorch.optim.lr_schedulerd, config['scheduler'])
    return cls(optimizer, **get_kwargs_for(cls, config, "scheduler"))

def format_for_hparams(config):
    out = {}
    for k, v in config.items():
        if isinstance(v, (list, tuple, set)):
            v = torch.tensor(list(v)) if all(isinstance(i, (int, float, bool)) for i in v) else np.array(list(v), dtype=object)
        elif isinstance(v, np.ndarray):
            v = torch.tensor(v) if np.issubdtype(v.dtype, np.number) else v
        out[k] = v
    return out

def save_hparams(writer, config, metric_dict): #metric_dict={'Epoch-correct/valid': 0}
    exp, ssi, sei = hparams(hparam_dict = format_for_hparams(config), metric_dict=metric_dict)   
    writer.file_writer.add_summary(exp)                 
    writer.file_writer.add_summary(ssi)                 
    writer.file_writer.add_summary(sei)
    return writer