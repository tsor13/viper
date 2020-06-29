import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical, Normal, Beta
from tqdm import tqdm
#from time import time
import pdb

from ViperEnvironment import ViperEnvironment
from autoencoder import Encoder, Decoder

device = 'cuda'

# initialize environment
model_path = 'model.pth'
hidden_size = 256
hidden_layers = 3
env = ViperEnvironment()
state_size = env.state_size
action_size = env.action_size
action_high = env.action_max
action_low = env.action_min
max_actions = np.inf
repeat = 25
gamma = .95

obs = env.reset()
# running mean
running_mean = torch.Tensor(obs).to(device)
# running var
running_var = torch.zeros(state_size).to(device) + .01
running_k = 1

# classes

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(hidden_layers)])
        self.linear2 = nn.Linear(hidden_size, action_size*2)
        self.sigmoid = nn.Sigmoid()

        self.action_size = action_size

        def init_weights(m):
            if type(m) == nn.Linear:
                if m.out_features == action_size*2:
                    e = 1e-3
                    m.weight.data.uniform_(-e, e)
                    # we want results close to 1 for initial alpha and beta
                    m.bias.data.uniform_(-e, e)
        self.apply(init_weights)


    def forward(self, x):
        x = self.linear1(x)
        x = self.hidden(x)
        x = self.linear2(x)
        x = x.reshape(-1, self.action_size, 2)
        x = self.sigmoid(x)
        dist = Normal(x[:,:,0], x[:,:,1]/2)
        return dist


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                 nn.ReLU(),
                                 *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(hidden_layers)],
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.policy_net = PolicyNetwork(64, action_size)
        self.value_net = ValueNetwork(64)
        
        encoder_path = 'encoder-relu.pth'
        encoder = Encoder()
        encoder.load_state_dict(torch.load(encoder_path))
        # freeze encoder
        for p in encoder.parameters():
            p.requires_grad = False
        self.encoder = encoder

    def forward(self, x):
        # run through network
        x = self.encoder(x)
        dist = self.policy_net(x)
        v = self.value_net(x)
        return dist, v


# dataset class
class RolloutDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        # TODO - why this instead of copying the entire dataset?
        self.data = []
        for d in data:
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# rollout functions
def get_action(model, state, greedy=False):
    with torch.no_grad():
        model.eval()
        x = torch.Tensor(state).float()
        x = x.to(device)
        global running_mean, running_var, running_k
        running_var = running_var + (x - running_mean) * (x - running_mean)
        running_mean = running_mean + (x - running_mean) / running_k
        running_k += 1
        x = x.unsqueeze(0)
        # normalize
        dist, v = model(x)
        # sample
        choice = dist.rsample()
        choice = torch.clamp(choice, 0, 1)
        p = dist.log_prob(choice).cpu().numpy()[0]
        choice = choice.cpu().numpy()[0]
        # scale to correct amount
        choice = choice * action_high + (1-choice) * action_low
        return choice, p

def rollout(model, render=False, greedy=False, train_reward=False):
    state = env.reset()
    done = False
    states, actions, probs, raw_rewards = [], [], [], []
    total_reward = 0
    while not done and len(actions) < max_actions:
        states.append(state)

        action, p = get_action(model, state, greedy)

        reward = 0
        for _ in range(repeat):
            state, r, done = env.step(action)
            reward += r
            if done:
                break

        actions.append(action)
        probs.append(p)
        raw_rewards.append(reward)
        total_reward += reward

    rewards = []
    R = 0
    while raw_rewards:
        R = raw_rewards.pop() + gamma*R
        rewards.insert(0, R)

    # TODO - memory correct?
    memory = [(s, a, p, r) for s, a, p, r in zip(states, actions, probs, rewards)]
    return memory, total_reward

def do_rollouts(model, n, train_reward=True):
    memory = []
    total_rewards = []
    for _ in tqdm(range(n)):
        m, r = rollout(model, train_reward=train_reward)
        memory.extend(m)
        total_rewards.append(r)
    return memory, total_rewards

def evaluate(model, n=100, greedy=False):
    total_rewards = []
    for _ in tqdm(range(n)):
        _, r = rollout(model, greedy=greedy, train_reward=False)
        total_rewards.append(r)
    print('Num episodes: {}\nMean: {:.3f}\nMean Standard Deviation: {:.3f}'.format(n, np.mean(total_rewards), np.std(total_rewards)/np.sqrt(n)))

def simulate(model):
    with open('output.txt', 'w') as file:
        done = False
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, p = get_action(model, state, False)


            reward = 0
            for _ in range(repeat):
                combined = np.hstack([state, action])
                row = ' '.join([str(f) for f in combined])
                file.write(row)
                file.write('\n')
                state, r, done = env.step(action)
                total_reward += r
                if done:
                    break
        file.close()
        print(total_reward)
        return total_reward

def train(params):
    model = params['model']
    model_path = params['model path']

    training_steps = params.get('training steps', 50)
    num_rollouts = params.get('num rollouts', 50)
    epsilon = params.get('epsilon', .2)

    learning_rate = params.get('learning rate', 3e-4)
    epochs = params.get('epochs', 5)
    batch_size = params.get('batch size', 16)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loop = tqdm(total = training_steps)

    best_mean_so_far = -np.inf

    train_reward = False

    linspace = torch.Tensor(np.linspace(0, 1, 1000))

    for step in range(0, training_steps):
        memory, total_rewards = do_rollouts(model, num_rollouts, train_reward=train_reward)
        dataset = RolloutDataset(memory)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        quantiles = [np.quantile(total_rewards, q) for q in np.linspace(0, 1, 5)]

        description = 'step: {}, [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]'.format(step, *quantiles)
        print(description)
        # loop.set_description(description)
        loop.update(1)

        if np.mean(total_rewards) > best_mean_so_far:
            best_mean_so_far = np.mean(total_rewards)
            torch.save(model.state_dict(), model_path)
            print('Saved!')

        value_losses = []
        policy_losses = []
        weight_losses = []
        entropies = []

        explained_variances = []
        policy_vars = []

        model.train()

        for epoch in range(epochs):
            for s, a, p, r in loader:
                s = s.float()
                s, a, p, r = s.to(device), a.to(device), p.to(device), r.to(device)
                dist, v = model(s)

                # VALUE LOSS
                r, v = r.flatten(), v.flatten()
                value_loss = (r - v).pow(2).mean()
                # value_loss *= .001

                # explained variance
                explained_variance = 1 - (r - v).var() / r.var()
                explained_variances.append(explained_variance.detach().item())

                # policy var
                policy_var = dist.loc.var()
                policy_vars.append(policy_var.item())

                # POLICY LOSS
                A = (r - v).detach().unsqueeze(1)
                n = len(a)
                indices = [i for i in range(n)]
                # map back to 0, 1
                a = (a - action_low) / (action_high - action_low)
                p_new = dist.log_prob(a)

                ratio = (p_new - p).exp()
                ratio = torch.clamp(ratio, .2, 5)
                term1 = ratio * A
                # sometimes the ratio is infinite due to floating point errors in large policy updates
                term1[term1 == -np.inf] = 0
                term2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * A
                term1, term2 = term1.unsqueeze(0), term2.unsqueeze(0)
                policy_loss = -torch.min(torch.cat([term1, term2]), 0)[0]
                policy_loss = policy_loss.mean()

                # ENTROPY LOSS
                entropy = dist.entropy().mean()
                # estimate entropy
                # linspaces = linspace.reshape(-1, 1, 1).repeat(1, n, action_size)
                # log_probs = dist.log_prob(linspaces)
                # entropy_mc = -(log_probs * log_probs.exp()).mean()
                # pdb.set_trace()
                # p0 = dist.cdf(torch.zeros(dist.mean.shape))
                # entropy_mc += -(torch.log(p0) * p0).mean()
                # p1 = 1-dist.cdf(torch.ones(dist.mean.shape))
                # entropy_mc += -(torch.log(p1) * p1).mean()
                # entropy *= .001
                # entropy *= .1
                # entropy *= 0
                # entropy = torch.log(dist.concentration0).pow(4).mean()
                # entropy += torch.log(dist.concentration1).pow(4).mean()
                # entropy *= .01

                # loss = value_loss + policy_loss - entropy
                loss = value_loss + policy_loss

                entropies.append(entropy.item())
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                prev_loss = loss.item()

                # L2 WEIGHT REGULARIZATION
                # for param in model.parameters():
                #     loss += .00001 * param.norm(2)

                weight_losses.append(loss.item() - prev_loss)
                # print('Weight: {:.6f}'.format(loss.item() - prev_loss))
                # if policy_loss.item() == np.inf:
                #     pdb.set_trace()
                #     pass

                optimizer.zero_grad()
                loss.backward()
                # clip the gradients, prevent exploding gradient
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                # check if param is nan
                for param in model.parameters():
                    if param.sum() != param.sum():
                        pdb.set_trace()
                        pass

        print('Entropy: {:.3}'.format(np.mean(entropies)))
        print('Policy Loss: {:.3}'.format(np.mean(np.abs(policy_losses))))
        print('Value Loss: {:.3}'.format(np.mean(value_losses)))
        print('Weight Loss: {:.3}'.format(np.mean(weight_losses)))
        print('Explained Variance: {:.3}'.format(np.mean(explained_variances)))
        print('Policy Var: {:.3}'.format(np.mean(policy_vars)))
        # mem, rew = rollout(model, True, False)
        # print(rew)
    return best_mean_so_far

model = PPONetwork(state_size, action_size)
model = model.to(device)
# model.load_state_dict(torch.load(model_path))
params = {
    'model': model,
    'model path': model_path,
    'epsilon': .2,
    'learning rate': 1e-5,
    'training steps': 80,
    'num rollouts': 20,
    'epochs': 20,
    'batch size': 64,
}
train(params)
# simulate(model)
# evaluate(model, 10)
