import math
import random
import torch
import torch.nn as nn
from torch.optim import SGD

π = math.pi

class Truck:
    def __init__(self, display=False):
        self.W = 1
        self.L = 1 * self.W
        self.d = 4 * self.L
        self.s = -0.1
        self.display = display
        self.box = [0, 40, -10, 10]
        
        if self.display:
            import matplotlib.pyplot as plt
            self.f = plt.figure(figsize=(10, 5), num='The truck backer-upper', facecolor='none')
            self.ax = self.f.add_axes([0.01, 0.01, 0.98, 0.98], facecolor='black')
            self.patches = list()
            self.ax.axis('equal')
            b = self.box
            self.ax.axis([b[0] - 1, b[1], b[2], b[3]])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.axhline()
            self.ax.axvline()

        self.reset()
    
    def reset(self, ϕ=0):
        self.ϕ = ϕ
        self.θ0 = random.random() * 2 * π
        self.θ1 = (random.random() - 0.5) * π / 2 + self.θ0
        self.x = (random.random() * .75 + 0.25) * self.box[1]
        self.y = (random.random() - 0.5) * (self.box[3] - self.box[2])
        if not self.valid():
            self.reset(ϕ)
        if self.display:
            self.draw()
    
    def step(self, ϕ=0, dt=1):
        if self.is_jackknifed():
            return None
        if self.is_offscreen():
            return None
            
        self.ϕ = ϕ
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        self.x += s * math.cos(θ0) * dt
        self.y += s * math.sin(θ0) * dt
        self.θ0 += s / L * math.tan(ϕ) * dt
        self.θ1 += s / d * math.sin(θ0 - θ1) * dt
        
        return (self.x, self.y, self.θ0, *self._traler_xy(), self.θ1)
    
    def state(self):
        return (self.x, self.y, self.θ0, *self._traler_xy(), self.θ1)
    
    def _get_atributes(self):
        return (self.x, self.y, self.W, self.L, self.d, self.s, self.θ0, self.θ1, self.ϕ)
    
    def _traler_xy(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        return x - d * math.cos(θ1), y - d * math.sin(θ1)
        
    def is_jackknifed(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        return abs(θ0 - θ1) * 180 / π > 90
    
    def is_offscreen(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        x1, y1 = x + 1.5 * L * math.cos(θ0), y + 1.5 * L * math.sin(θ0)
        x2, y2 = self._traler_xy()
        b = self.box
        return not (b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3])
        
    def valid(self):
        return not self.is_jackknifed() and not self.is_offscreen()

    def draw(self):
        if not self.display: return
        if self.patches: self.clear()
        self._draw_car()
        self._draw_trailer()
        self.f.canvas.draw()
            
    def clear(self):
        for p in self.patches:
            p.remove()
        self.patches = list()
        
    def _draw_car(self):
        import matplotlib
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        ax = self.ax
        
        x1, y1 = x + L / 2 * math.cos(θ0), y + L / 2 * math.sin(θ0)
        bar = Line2D((x, x1), (y, y1), lw=5, color='C2', alpha=0.8)
        ax.add_line(bar)

        car = Rectangle(
            (x1, y1 - W / 2), L, W, color='C2', alpha=0.8, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x1, y1, θ0 * 180 / π) +
            ax.transData
        )
        ax.add_patch(car)

        x2, y2 = x1 + L / 2 ** 0.5 * math.cos(θ0 + π / 4), y1 + L / 2 ** 0.5 * math.sin(θ0 + π / 4)
        left_wheel = Line2D(
            (x2 - L / 4 * math.cos(θ0 + ϕ), x2 + L / 4 * math.cos(θ0 + ϕ)),
            (y2 - L / 4 * math.sin(θ0 + ϕ), y2 + L / 4 * math.sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(left_wheel)

        x3, y3 = x1 + L / 2 ** 0.5 * math.cos(π / 4 - θ0), y1 - L / 2 ** 0.5 * math.sin(π / 4 - θ0)
        right_wheel = Line2D(
            (x3 - L / 4 * math.cos(θ0 + ϕ), x3 + L / 4 * math.cos(θ0 + ϕ)),
            (y3 - L / 4 * math.sin(θ0 + ϕ), y3 + L / 4 * math.sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(right_wheel)
        
        self.patches += [car, bar, left_wheel, right_wheel]
        
    def _draw_trailer(self):
        import matplotlib
        from matplotlib.patches import Rectangle
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        ax = self.ax
            
        x, y = x - d * math.cos(θ1), y - d * math.sin(θ1) - W / 2
        trailer = Rectangle(
            (x, y), d, W, color='C0', alpha=0.8, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x, y + W/2, θ1 * 180 / π) +
            ax.transData
        )
        ax.add_patch(trailer)
        
        self.patches += [trailer]

def generate_truck_data(episodes=10):
    """
    Generates training data for the emulator network.
    """
    truck = Truck()
    inputs = []
    outputs = []
    for episode in range(episodes):
        truck.reset()
        while truck.valid():
            initial_state = truck.state()
            ϕ = (random.random() - 0.5) * π / 2
            step_output = truck.step(ϕ)
            if step_output is not None:
                inputs.append((ϕ, *initial_state))
                outputs.append(step_output)
    return inputs, outputs

def model_trucker(train_inputs, train_outputs, state_size=6, steering_size=1, hidden_units=45, lr=0.005, epochs=1):
    """
    Train the emulator network to predict the next state given steering input and current state.
    train_inputs: Tensor of shape (N, steering_size + state_size)
    train_outputs: Tensor of shape (N, state_size)
    """
    emulator = nn.Sequential(
        nn.Linear(steering_size + state_size, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, state_size)
    )
    optimiser_e = SGD(emulator.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for i in torch.randperm(len(train_inputs)):
            ϕ_state = train_inputs[i]
            next_state_prediction = emulator(ϕ_state)
            
            next_state = train_outputs[i]
            loss = criterion(next_state_prediction, next_state)
            
            optimiser_e.zero_grad()
            loss.backward()
            optimiser_e.step()
            
    return emulator

def trucker_call(emulator, test_inputs):
    """
    Perform prediction using the trained emulator.
    test_inputs: Tensor of shape (N, steering_size + state_size)
    """
    emulator.eval()
    predictions = []
    with torch.no_grad():
        for ϕ_state in test_inputs:
            predictions.append(emulator(ϕ_state))
    return torch.stack(predictions)
