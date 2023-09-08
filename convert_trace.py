
import torch

from agent import SAC


path = "./store/trace-CartPole-v0-logbeta-5.0-step93-perf76.0-actor.pth"
model_dicts = torch.load(path)

agent = SAC(state_size=4,
            action_size=2,
            device="cpu",
            log_beta=None)

agent.actor_local.load_state_dict(model_dicts)
example = torch.zeros((1, 4), device="cpu").float()
tr = torch.jit.trace(agent.actor_local, example)
torch.jit.save(tr, path[:-1])
