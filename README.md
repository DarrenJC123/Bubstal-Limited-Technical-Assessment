# Bubstal-Limited-Technical-Assessment
## Task 1:
This algorithm uses a Naive Bayes probability‑based classifier that predicts whether a customer’s message expresses Happy or Frustrated sentiment and uses Laplace smoothing to handle unseen words.

### 1. Training Setup
- Tokenization & Standardisation
  - Convert text to lowercase.
  - Remove basic punctuation (. and ,).
  - Split on whitespace to obtain words.

- Vocabulary Counting
  - For each class (Happy / Frustrated), count the frequency of every unique word found in the training data.
  - Record the total number of words per class.
 
- Compute Prior Probabilities (i.e., proportion of happy and frustrated in the whole training set)
  
  $P(Happy) = \frac{\\#Happy Samples}{Total Samples}$
  $P(Frustrated) = \frac{\\#Frustrated Samples}{Total Samples}$

- Conditional Probabilities (with Laplace Smoothing)
To avoid zero probabilities for unseen words, we add a small pseudo‑count to every possible word.
The smoothed probability of a word $w$ given class $c$ is:

  $P(w|c) = \frac{count(w,c) + 1}{total words in class c + V}$

  Where $V$ is the size of the global vocabulary (all unique words seen in training across both classes).

### 2. Prediction Phase
For a new sentence:
  - Tokenise and standardise exactly as in training.
  - Compute unnormalised posterior scores:

    $score_c = P(c) \times \prod_{w \in test words} P(w|c)$

  - Return the class with the higher score, together with its score (which is proportional to the true posterior probability).

### Time Complexity
Let 
- $n$ = number of training samples
- $L$ = average number of words per training sample
- $V$ = total distinct words in the training vocabulary
- $m$ = number of words in a test sentence

Training
- Tokenisation and counting – $O(n⋅L)$
  Each word is processed once.
  Building vocabulary and frequency tables – also $O(n⋅L).$
  Computing priors & storing counts – constant time per class.

- Overall training complexity: $O(n⋅L)$.
  Because $n⋅L$ equals the total number of word tokens in the training set, this is linear in the data size.
  
- Prediction (per test sentence)
  Tokenisation – $O(m)$
  
- Score computation – For each word, we perform a dictionary lookup (average $O(1)$ ) and a multiplication.
  Hence $O(m)$ per test sentence.

### Memory Usage
  Two dictionaries (one per class), each storing at most $O(V)$ space.
  $V$ is bounded by the total number of unique tokens in the training data.

## Notes On Output:
The classifier achieves only 50% accuracy on the test dataset due to unseen word bias in the small training sample.

### Example: Test Case 1 (Expected: Happy): "The delivery was incredibly quick and the item is brilliant!"
The Words "quick" and "item" exists in Frustrated training data but not in Happy training data

This single word biases the prediction toward Frustrated, overriding all happy signals

### Solution
Accuracy improves significantly with more training data because:
- Words become more concentrated in the correct class
- The model learns that words like "quick" appears more in the Happy class
- Smoothing becomes less aggressive as vocabulary coverage increases

## Task 2: The Optimizer
### Algorithm Overview
- Cost & Gradient Functions
cost(x) = $x^2 - 6x + 9$ – the function we want to minimize, gradient(x) = $2x - 6$ – its derivative.

- Gradient Descent Loop
For each iteration:
  1. Compute the gradient at the current x.
  2. Update x_new = x_old - learning_rate * gradient.
  3. Apply a small learning rate decay (learning_rate *= 0.999) to improve convergence stability.
  4. If the absolute change |x_new - x_old| falls below a tolerance (1e-6), stop early – the solution has converged.
  5. Print intermediate values at specific iterations (1, 10, 100, …) for monitoring.

The loop continues until either the maximum number of iterations is reached or the early-stopping condition is met.

- Return Value
The function returns the final x that minimises the cost.

### Time Complexity
Let $n$ = number of iterations actually performed (after early stopping).
- Each iteration does: One gradient evaluation: $O(1)$
- One update calculation: $O(1)$
- One cost evaluation (for printing): $O(1)$

Therefore, the total time complexity is: $O(n)$

### Memory usage
$O(1)$ (Fixed Variables)
