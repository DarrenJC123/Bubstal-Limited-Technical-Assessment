def cost(x):
    return x*x - 6*x + 9

def gradient(x):
    return 2*x - 6

def gradient_descent(initial_x, learning_rate, num_iterations):
    x_next = float('inf')
    x_prev = initial_x
    for i in range(num_iterations):
        grad = gradient(x_prev)
        x_next = x_prev - learning_rate * grad
        learning_rate *= 0.999  # Decay learning rate
        if abs((x_next - x_prev)) < 1e-6:  # Early Stopping
            break
        x_prev = x_next
        if i in [1, 10, 100, 1000, 5000, 9999]:
            print(f"Iteration {i}: x = {x_next}, cost = {cost(x_next)}")
    return x_next

print(gradient_descent(initial_x=0, learning_rate=0.99, num_iterations=250)) # Converge from the left
print(gradient_descent(initial_x=3, learning_rate=0.99, num_iterations=250)) # Ensure proper early stopping
print(gradient_descent(initial_x=6, learning_rate=0.99, num_iterations=250)) # Converge from the right