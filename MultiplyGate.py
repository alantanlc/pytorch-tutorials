class MultiplyGate(object):

    def forward(self, x, y):
        z = x * y
        self.x = x # must keep this around!
        self.y = y  # must keep this around!
        return z

    def backward(self, dz): # dL / dz
        dx = self.y * dz    # [dz/dx * dL/dz]
        dy = self.x * dz    # [dz/dy * dL/dz]
        return [dx, dy] # dL / dx