def cost(x):
    return x*x - 6*x + 9

def gradient(x):
    return 2*x - 6

def gradient_descent(initial_x, learning_rate, num_iterations):
    x_next = float('inf')
    x_prev = initial_x
    for _ in range(num_iterations):
        grad = gradient(x_prev)
        x_next = x_prev - learning_rate * grad
        learning_rate *= 0.999  # Decay learning rate
        if abs((x_next - x_prev)) < 1e-6:  # Early Stopping
            break
        x_prev = x_next
    
    return x_next

print(gradient_descent(initial_x=0, learning_rate=0.99, num_iterations=100000))