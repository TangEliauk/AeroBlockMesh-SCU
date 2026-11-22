"""Continuous Soft Actor-Critic with twin target critics, implemented in NumPy."""

import numpy as np


class MLP:
    def __init__(self, widths, rng, small_output=False):
        self.weights = []
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            scale = 0.02 if small_output and index == len(widths) - 2 else (2 / inputs) ** 0.5
            self.weights.extend([
                (rng.standard_normal((inputs, outputs)) * scale).astype(np.float32),
                np.zeros(outputs, dtype=np.float32),
            ])
        self.first = [np.zeros_like(weight) for weight in self.weights]
        self.second = [np.zeros_like(weight) for weight in self.weights]
        self.updates = 0

    def forward(self, inputs):
        layers = [inputs]
        for index in range(0, len(self.weights), 2):
            value = layers[-1] @ self.weights[index] + self.weights[index + 1]
            if index + 2 < len(self.weights):
                value = np.maximum(value, 0)
            layers.append(value)
        return layers[-1], layers

    def backward(self, layers, gradient, update=True, learning_rate=3e-4):
        gradients = [None] * len(self.weights)
        for index in reversed(range(len(layers) - 1)):
            offset = 2 * index
            if index + 2 < len(layers):
                gradient = gradient * (layers[index + 1] > 0)
            if update:
                gradients[offset] = layers[index].T @ gradient
                gradients[offset + 1] = gradient.sum(axis=0)
            gradient = gradient @ self.weights[offset].T
        if update:
            self.adam(gradients, learning_rate)
        return gradient

    def adam(self, gradients, learning_rate):
        self.updates += 1
        correction1 = 1 - 0.9 ** self.updates
        correction2 = 1 - 0.999 ** self.updates
        for weight, first, second, gradient in zip(self.weights, self.first, self.second, gradients):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient ** 2
            weight -= (learning_rate / correction1) * first / (np.sqrt(second / correction2) + 1e-7)

    def track(self, source, coefficient):
        for target, online in zip(self.weights, source.weights):
            target += coefficient * (online - target)


class SAC:
    def __init__(self, observation_dim, action_dim, rng, width=64, alpha=0.035):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.width = width
        self.alpha = alpha
        self.rng = rng
        self.actor = MLP((observation_dim, width, width, 2 * action_dim), rng, small_output=True)
        self.actor.weights[-1][action_dim:] = -1.45
        critic_widths = (observation_dim + action_dim, width, width, 1)
        self.critics = [MLP(critic_widths, rng, small_output=True) for _ in range(2)]
        self.targets = [MLP(critic_widths, rng, small_output=True) for _ in range(2)]
        for target, critic in zip(self.targets, self.critics):
            target.track(critic, 1)

    def action(self, observation, deterministic=False):
        output, _ = self.actor.forward(observation[None, :])
        if deterministic:
            return np.tanh(output[0, :self.action_dim])
        return self.sample(output)[0][0]

    def sample(self, output):
        mean, raw_log_std = np.split(output, 2, axis=1)
        log_std = np.clip(raw_log_std, -4.5, 0.75)
        noise = self.rng.standard_normal(mean.shape, dtype=np.float32)
        perturbation = np.exp(log_std) * noise
        raw_action = mean + perturbation
        action = np.tanh(raw_action)
        log_det = 2 * (np.float32(np.log(2)) - raw_action - np.logaddexp(0, -2 * raw_action))
        log_prob = (-0.5 * noise ** 2 - log_std - np.float32(0.5 * np.log(2 * np.pi)) - log_det).sum(
            axis=1, keepdims=True
        )
        return action, log_prob, perturbation, (log_std == raw_log_std)

    def imitate(self, obs, actions, weight=1.0, learning_rate=1e-3):
        output, layers = self.actor.forward(obs)
        mean_action = np.tanh(output[:, :self.action_dim])
        gradient = np.zeros_like(output)
        gradient[:, :self.action_dim] = (
            (2 * weight / mean_action.size) * (mean_action - actions) * (1 - mean_action ** 2)
        )
        self.actor.backward(layers, gradient, learning_rate=learning_rate)

    def update(self, replay, batch_size=64, update_actor=True):
        obs, actions, reward, next_obs, terminal = replay.batch(self.rng, batch_size)
        next_output, _ = self.actor.forward(next_obs)
        next_actions, next_log_prob, _, _ = self.sample(next_output)
        next_inputs = np.concatenate((next_obs, next_actions), axis=1)
        next_q = np.minimum(*(network.forward(next_inputs)[0] for network in self.targets))
        target_q = reward + (1 - terminal) * 0.98 * (next_q - self.alpha * next_log_prob)

        critic_inputs = np.concatenate((obs, actions), axis=1)
        for critic in self.critics:
            prediction, layers = critic.forward(critic_inputs)
            gradient = (2 / batch_size) * np.clip(prediction - target_q, -3, 3)
            critic.backward(layers, gradient)

        if update_actor:
            output, actor_layers = self.actor.forward(obs)
            sampled_actions, _, perturbation, active_std = self.sample(output)
            actor_inputs = np.concatenate((obs, sampled_actions), axis=1)
            (q1, layers1), (q2, layers2) = (critic.forward(actor_inputs) for critic in self.critics)
            first_minimum = (q1 <= q2).astype(np.float32)
            dq1 = self.critics[0].backward(layers1, first_minimum, update=False)
            dq2 = self.critics[1].backward(layers2, 1 - first_minimum, update=False)
            dq_da = (dq1 + dq2)[:, -self.action_dim:]
            mean_gradient = 2 * self.alpha * sampled_actions - dq_da * (1 - sampled_actions ** 2)
            std_gradient = (mean_gradient * perturbation - self.alpha) * active_std
            gradient = np.concatenate((mean_gradient, std_gradient), axis=1) / batch_size
            self.actor.backward(actor_layers, gradient)

        for target, critic in zip(self.targets, self.critics):
            target.track(critic, 0.01)

    def save_actor(self, path):
        np.savez_compressed(
            path, observation_dim=self.observation_dim, action_dim=self.action_dim,
            width=self.width, **{f"layer_{i}": array for i, array in enumerate(self.actor.weights)}
        )

    def load_actor(self, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            for index, target in enumerate(self.actor.weights):
                source = checkpoint[f"layer_{index}"]
                if source.shape != target.shape:
                    raise ValueError(f"Actor layer {index} shape does not match")
                target[...] = source
