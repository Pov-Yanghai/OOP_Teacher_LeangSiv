# Exercise 1.1: The Interpreter
# Running these statements in the Python interpreter
print(3 + 1)      # 4
print(3 * 3)      # 9
print(2 ** 3)     # 8
print("Hello, world!")  # String output

# Exercise 1.2: Scripts
# 

# Exercise 1.3: More Interpreter
py = "py"
thon = "thon"
print(py + thon)     #'python'
print(py * 3 + thon) # 'pypypython'
#print(py - py)      # Error
##print('3' + 3)         # Error
print(3 * '3')         # 333
#a                 # NameError if 'a' is not defined
#a = 3             # Assigning 3 to variable 'a'
#print(a)          # Output: 3

# Exercise 1.4: Booleans
print(1 == 1)         # True
print(1 == True)      # True
print(0 == True)      # False
print(0 == False)     # True
print(3 == 1 * 3)     # True
print((3 == 1) * 3)   # 0 (False * 3)
print((3 == 3) * 4 + 3 == 1)  # False
print(3**5 >= 4**4)   # False

# Exercise 1.5: Integers
print(5 / 3)          # 1.666... 
print(5 % 3)          # 2 (remainder of division)
print(5.0 / 3)        # 1.666...
print(5 / 3.0)        # 1.666...
print(5.2 % 3)        # 2.2 (modulus with float)
#print(2001 ** 200)   # Very large number, cause long processing time

# Exercise 1.6: Floats
#print(2000.3 ** 200) # OverflowError: too large for float
print(1.0 + 1.0 - 1.0)  # 1.0
print(1.0 + 1.0e20 - 1.0e20)  # 0.0 

# Exercise 1.7: Variables
name = "John Doe"
print(f"Hello, {name}!")

# Exercise 1.8: Type Casting
print(float(123))         # 123.0
print(float(123.23))      # 123.23
print(int(123.23))        # 123 
print(int(float(123.23))) # 123
print(str(12))            # '12'
print(str(12.2))          # '12.2'
a = 10
print(bool(a))            # True
print(bool(0))            # False
print(bool(0.1))          # True
# Exercise 2.1: Range
print(range(5))  # In Python 3, this returns a range object
print(list(range(5)))  # Convert range to list to see the values
### ------------------------------------------------------------------------------
## PArt 2 
# Exercise 2.2: For loops
# (a) Print numbers 0 to 100
for i in range(101):
    print(i)

# (b) Print numbers 0 to 100 divisible by 7
for i in range(0, 101, 7):
    print(i)

# (c) Print numbers 1 to 100 divisible by 5 but not by 3
for i in range(1, 101):
    if i % 5 == 0 and i % 3 != 0:
        print(i)

# (d) Print divisors of numbers from 2 to 20 (excluding 1 and itself)
for x in range(2, 21):
    divisors = [i for i in range(2, x) if x % i == 0]
    print(f"Divisors of {x}: {divisors}")

# Exercise 2.3: While loops
# (a) Print numbers 0 to 100
x = 0
while x <= 100:
    print(x)
    x += 1

# (b) Print numbers 0 to 100 divisible by 7
x = 0
while x <= 100:
    if x % 7 == 0:
        print(x)
    x += 1

# Exercise 2.5: Find first 20 numbers divisible by 5, 7, and 11
count = 0
x = 11
while count < 20:
    if x % 5 == 0 and x % 7 == 0 and x % 11 == 0:
        print(x)
        count += 1
    x += 1

# Exercise 2.6: Smallest number divisible by all integers from 1 to 10
import math
from functools import reduce

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

smallest_number = reduce(lcm, range(1, 11))
print("Smallest number divisible by all integers from 1 to 10:", smallest_number)

# Exercise 2.7: Collatz sequence
def collatz_sequence(start):
    while start != 1:
        print(start, end=' ')
        start = start // 2 if start % 2 == 0 else 3 * start + 1
    print(1)

collatz_sequence(103)

# Exercise 3.1: Hello
def hello_world():
    print("Hello, world!")

def hello_name(name):
    print(f"Hello, {name}!")

# The difference between print and return is that print outputs to the console, while return gives a value back to be used.

# Exercise 3.2: Polynomial
def polynomial(x):
    return 3 * x**2 - x + 2

# Exercise 3.3: Maximum
def my_max(x, y):
    if x > y:
        return x
    else:
        return y

def my_max__butno_else(x, y):
    if x > y:
        return x
    return y

# Exercise 3.4: Primes
def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def primes_up_to(n):
    return [x for x in range(2, n + 1) if is_prime(x)]

def first_n_primes(n):
    primes = [] ## list of primes
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

# Exercise 3.5: Root Finding
def root(f, a, b, tol=1e-7):
    if f(a) * f(b) > 0:
        print("Function evals have same sign")
        return None
    while abs(b - a) > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# Exercise 4.1: Short questions
# (a) Print elements of a list
def print_elements(my_list):
    for item in my_list:
        print(item)

# (b) Print elements of a list in reverse
def print_reverse(my_list):
    for item in reversed(my_list):
        print(item)

# (c) Implement len function
def my_len(my_list):
    count = 0
    for item in my_list:
        count += 1
    return count


# Exercise 4.2: Copying lists
# (a) Create list a
a = [1, 2, 3, 4]

# (b) Set b = a
b = a

# (c) Change b[1]
b[1] = 10

# (d) a is also changed

# (e) Set c = a[:]
c = a[:]

# (f) Change c[2]
c[2] = 20

# (g) a remains unchanged

# Function to set first element to zero
def set_first_elem_to_zero(l):
    l[0] = 0
    return l


# Exercise 4.3: Lists of lists
# Difference between a and b
a = [[]] * 3
b = [[] for _ in range(3)]


# Exercise 4.4: Lists and functions
def set_at_index_to_zero(lst, index):
    lst[index] = 0
    return lst


# Exercise 4.5: Primes
def primes_up_to_n(n):
    primes = []
    for num in range(2, n + 1):
        is_prime = all(num % i != 0 for i in range(2, num))
        if is_prime:
            primes.append(num)
    return primes


def first_n_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % i != 0 for i in range(2, num)):
            primes.append(num)
        num += 1
    return primes


# Exercise 4.6: List comprehensions
# (a) Generate [i, j]
i, j = 1, 2
lst = [i, j]

# (b) Generate [i, j] where i < j
lst = [(i, j) for i in range(1, 5) for j in range(i + 1, 6)]

# (c) Generate i + j where i and j are prime and i > j
primes = [i for i in range(2, 20) if all(i % j != 0 for j in range(2, i))]
result = [i + j for i in primes for j in primes if i > j]

# (d) Polynomial evaluation
def evaluate_polynomial(x, coefs):
    return sum(coef * (x ** i) for i, coef in enumerate(coefs))


# Exercise 4.7: Filter
def myfilter(func, lst):
    return [item for item in lst if func(item)]


# Exercise 4.8: Flatten a list of lists
def flatten(lst):
    return [item for sublist in lst for item in sublist]


# Exercise 4.9: Finding the longest word
import string

def longest_word(text):
    words = text.translate(str.maketrans('', '', string.punctuation)).split()
    return max(words, key=len)


# Exercise 4.10: Collatz sequence, part 2
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence


def longest_collatz_sequence(limit):
    longest = 0
    max_length = 0
    for n in range(1, limit):
        length = len(collatz_sequence(n))
        if length > max_length:
            longest = n
            max_length = length
    return longest


# Exercise 4.11: Pivots
def pivot_list(x, ys):
    less_than_x = [y for y in ys if y < x]
    greater_than_x = [y for y in ys if y > x]
    return less_than_x + [x] + greater_than_x


# Exercise 4.12: Prime challenge
def primes(n):
    return list(filter(lambda x: all(x % i != 0 for i in range(2, x)), range(2, n + 1)))



#Tuples
# Exercise 5.1: Swapping two values
# Swapping two variables a and b in one line using tuple unpacking
a, b = 10, 20
a, b = b, a
print(a, b)  # Output: 20, 10

# Exercise 5.2: Zip
# (a) Create a list of coordinates (x, y) as tuples using zip
x = [1, 2, 3]
y = [4, 5, 6]
coordinates = list(zip(x, y))
print(coordinates)  # Output: [(1, 4), (2, 5), (3, 6)]

# (b) Unzip the list of tuples to get the two separate lists again
x_unzip, y_unzip = zip(*coordinates)
x_unzip = list(x_unzip)
y_unzip = list(y_unzip)
print(x_unzip)  # Output: [1, 2, 3]
print(y_unzip)  # Output: [4, 5, 6]

# Exercise 5.3: Distances

def l1_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))

# Example
x = (1, 2, 3)
y = (4, 5, 6)
print(l1_distance(x, y))  # Output: 9


import math

def l2_distance(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

# Example
x = (1, 2, 3)
y = (4, 5, 6)
print(l2_distance(x, y))  

## Dictionaries 


# Exercise 6.1: Printing a dictionary
def print_dict(my_dict):
    for key in my_dict:
        print(f"{key}: {my_dict[key]}")

# Example:
my_dict = {'a': 1, 'b': 2, 'c': 3}
print_dict(my_dict)  # Output: a: 1, b: 2, c: 3


# Exercise 6.2: Histogram
def histogram(lst):
    result = {}  ## dictionary 
    for item in lst:
        if item in result:
            result[item] += 1
        else:
            result[item] = 1
    return result

# Example:
lst = ['a', 'b', 'a', 'c', 'b', 'a']
print(histogram(lst))  # Output: {'a': 3, 'b': 2, 'c': 1}


# Exercise 6.3: Get method
def histogram_get(lst):
    result = {}
    for item in lst:
        result[item] = result.get(item, 0) + 1
    return result

# Example:
lst = ['a', 'b', 'a', 'c', 'b', 'a']
print(histogram_get(lst))  # Output: {'a': 3, 'b': 2, 'c': 1}


# Exercise 6.4: Random text generator
import random

# (a) process_line function
def process_line(line):
    words = line.split()
    transitions = []
    for i in range(len(words)-1):
        transitions.append((words[i], words[i+1]))
    return transitions

# Example:
line = "the fire and the wind"
print(process_line(line))  # Output: [('the', 'fire'), ('fire', 'and'), ('and', 'the'), ('the', 'wind')]

# (b) process_textfile function
def process_textfile(f):
    transitions = {}
    for line in f:
        for current_state, next_state in process_line(line):
            if current_state not in transitions:
                transitions[current_state] = []
            transitions[current_state].append(next_state)
    return transitions

# Example:
f = ["the fire and the wind", "and the rain fell"]
print(process_textfile(f))
# Output: {'the': ['fire', 'wind'], 'fire': ['and'], 'and': ['the', 'rain'], 'wind': ['and'], 'rain': ['fell']}

# (c) generate_with_transitions function
def generate_with_transitions(transitions):
    current_state = 'BEGIN'
    sentence = []
    while current_state != 'END':
        next_state = random.choice(transitions.get(current_state, ['END']))
        sentence.append(next_state)
        current_state = next_state
    return ' '.join(sentence)

# Example:
transitions = {'BEGIN': ['the'], 'the': ['fire', 'wind'], 'fire': ['and'], 'wind': ['the'], 'and': ['the'], 'END': []}
print(generate_with_transitions(transitions))  # Output: 'the fire and the END'


# Exercise 6.5: Vector functions
# (a) Add two (dense) vectors
def add_dense_vectors(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(v1[i] + v2[i])
    return result

# Example:
v1 = [1, 2, 3]
v2 = [4, 5, 6]
print(add_dense_vectors(v1, v2))  # Output: [5, 7, 9]

# (b) Multiply two (dense) vectors (inner product)
def multiply_dense_vectors(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

# Example:
print(multiply_dense_vectors(v1, v2))  # Output: 32

# (c) Add two sparse vectors
def add_sparse_vectors(v1, v2):
    result = {}
    for key in v1:
        result[key] = v1[key]
    for key in v2:
        if key in result:
            result[key] += v2[key]
        else:
            result[key] = v2[key]
    return result

# Example:
v1 = {0: 1, 2: 4}
v2 = {1: 2, 2: 3}
print(add_sparse_vectors(v1, v2))  # Output: {0: 1, 1: 2, 2: 7}

# (d) Multiply two sparse vectors (inner product)
def multiply_sparse_vectors(v1, v2):
    result = 0
    for key in v1:
        if key in v2:
            result += v1[key] * v2[key]
    return result

# Example:
print(multiply_sparse_vectors(v1, v2))  # Output: 12

# (e) Add a sparse vector and a dense vector
def add_sparse_dense(v_sparse, v_dense):
    result = v_sparse.copy()
    for i in range(len(v_dense)):
        if i in result:
            result[i] += v_dense[i]
        else:
            result[i] = v_dense[i]
    return result

# Example:
v_sparse = {0: 1, 2: 4}
v_dense = [1, 2, 3]
print(add_sparse_dense(v_sparse, v_dense))  # Output: {0: 2, 1: 2, 2: 7}

# (f) Multiply a sparse vector and a dense vector
def multiply_sparse_dense(v_sparse, v_dense):
    result = 0
    for i in range(len(v_dense)):
        if i in v_sparse:
            result += v_sparse[i] * v_dense[i]
    return result

# Example:
print(multiply_sparse_dense(v_sparse, v_dense))  # Output: 14


# Exercise 6.6: Reverse look-up
def reverse_lookup(my_dict, value):
    for key in my_dict:
        if my_dict[key] == value:
            return key
    return None

# Example:
my_dict = {'a': 1, 'b': 2, 'c': 3}
print(reverse_lookup(my_dict, 2))  # Output: 'b'


## File I/O 
# Exercise 7.1: Open a file
def open_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                print(line.strip())  # strip() removes   newlines at the end
    except FileNotFoundError:
        print("File not found!")

# Example:
# open_file("sample.txt")  # This will open and print the file line by line.

# Exercise 7.2: Wordcount

# (a) Find the 20 most common words
from collections import Counter

def word_count(filename):
    with open(filename, 'r') as file:
        words = file.read().lower().split()  # read all text, convert to lowercase, and split into words
        word_counts = Counter(words)
        return word_counts.most_common(20)  # return the 20 most common words

# Example:
# word_count("shakespeare.txt")  # This will print the 20 most common words from the file.


# (b) How many unique words are used?
def unique_word_count(filename):
    with open(filename, 'r') as file:
        words = file.read().lower().split()
        unique_words = set(words)  # using set to get unique words
        return len(unique_words)

# Example:
# print(unique_word_count("shakespeare.txt"))  # This will print the number of unique words.


# (c) How many words are used at least 5 times?
def words_used_at_least_five_times(filename):
    with open(filename, 'r') as file:
        words = file.read().lower().split()
        word_counts = Counter(words)
        return len([word for word, count in word_counts.items() if count >= 5])

# Example:
# print(words_used_at_least_five_times("shakespeare.txt"))  # This will print how many words are used at least 5 times.


# (d) Write the 200 most common words, and their counts, to a file.
def write_most_common_words_to_file(filename, output_file):
    with open(filename, 'r') as file:
        words = file.read().lower().split()
        word_counts = Counter(words)
        most_common = word_counts.most_common(200)
    
    with open(output_file, 'w') as file:
        for word, count in most_common:
            file.write(f"{word}: {count}\n")  # Write each word and its count to the file

# Example:
# write_most_common_words_to_file("shakespeare.txt", "common_words.txt")  # This writes the 200 most common words to a new file.


# Exercise 7.3: Random text generator II

# (a) Change the process_textfile function to read from a file
def process_textfile(filename):
    transitions = {}
    with open(filename, 'r') as file:
        for line in file:
            words = line.split()
            for i in range(len(words)-1):
                current_word = words[i]
                next_word = words[i+1]
                if current_word not in transitions:
                    transitions[current_word] = []
                transitions[current_word].append(next_word)
    return transitions

# Example:
# transitions = process_textfile("shakespeare.txt")
# print(transitions)  # This will print the transitions (word pairs) from the file.


# (b) Modify code to use two words instead of one (bigram model)
def process_textfile_bigram(filename):
    transitions = {}
    with open(filename, 'r') as file:
        for line in file:
            words = line.split()
            for i in range(len(words)-2):
                current_state = (words[i], words[i+1])  # use two words as state
                next_word = words[i+2]
                if current_state not in transitions:
                    transitions[current_state] = []
                transitions[current_state].append(next_word)
    return transitions

# Example:
# transitions = process_textfile_bigram("shakespeare.txt")
# print(transitions)  # This will print the bigram transitions from the file.


# Exercise 7.4: Sum of lists

# (a) Write a function that generates random data
import random

def generate_data(n, a, b, filename):
    with open(filename, 'w') as file:
        for _ in range(n):
            file.write(f"{random.randint(a, b)}\n")  # write n random numbers between a and b

# Example:
# generate_data(2000, 1, 10000, "data1.txt")  # Generates 2000 random numbers between 1 and 10000.


# (b) Reading the data
def read_data(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file]  # convert lines to integers

# Example:
# data1 = read_data("data1.txt")
# print(data1)  # This will print the data read from the file.


# (c) Find pairs u and v such that u + v = k
def find_pairs(filename1, filename2, k):
    list1 = read_data(filename1)
    list2 = read_data(filename2)
    pairs = []
    for u in list1:
        for v in list2:
            if u + v == k:
                pairs.append((u, v))
    return pairs

# Example:
# pairs = find_pairs("data1.txt", "data2.txt", 5000)
# print(pairs)  # This will print the pairs of numbers that add up to 5000.


# (d) Testing with sample data
generate_data(2000, 1, 10000, "data1.txt")
generate_data(2000, 1, 10000, "data2.txt")
pairs_5000 = find_pairs("data1.txt", "data2.txt", 5000)
pairs_12000 = find_pairs("data1.txt", "data2.txt", 12000)
print(pairs_5000)
print(pairs_12000)


# (e) Bonus: Efficient 
def find_pairs_efficient(filename1, filename2, k):
    list1 = read_data(filename1)
    list2 = read_data(filename2)
    seen = set(list2)
    pairs = []
    for u in list1:
        if k - u in seen:
            pairs.append((u, k - u))
    return pairs

# Example:
# pairs = find_pairs_efficient("data1.txt", "data2.txt", 5000)
# print(pairs)  # Efficient solution for the sum problem.

### Classes 
import math

# Exercise 8.1: Rational Numbers
class Rational:
    def __init__(self, p, q):
        if q == 0:
            raise ValueError("Denominator cannot be zero")
        gcd = math.gcd(p, q)
        self.p = p // gcd  # Simplify the fraction
        self.q = q // gcd
    
    def __str__(self):
        return f"{self.p}/{self.q}"
    
    def __repr__(self):
        return f"Rational({self.p}, {self.q})"
    
    def __add__(self, other):
        new_p = self.p * other.q + other.p * self.q
        new_q = self.q * other.q
        return Rational(new_p, new_q)
    
    def __sub__(self, other):
        new_p = self.p * other.q - other.p * self.q
        new_q = self.q * other.q
        return Rational(new_p, new_q)
    
    def __mul__(self, other):
        return Rational(self.p * other.p, self.q * other.q)
    
    def __truediv__(self, other):
        if other.p == 0:
            raise ValueError("Cannot divide by zero")
        return Rational(self.p * other.q, self.q * other.p)
    
    def __eq__(self, other):
        return self.p == other.p and self.q == other.q
    
    def __float__(self):
        return self.p / self.q
    
    def reciprocal(self):
        return Rational(self.q, self.p)  # Swap numerator and denominator

# Example usage:
r1 = Rational(10, 20)
r2 = Rational(1, 4)
print(r1)  # Output: 1/2
print(r2)  # Output: 1/4
print(r1 + r2)  # Output: 3/4
print(r1 - r2)  # Output: 1/4
print(r1 * r2)  # Output: 1/8
print(r1 / r2)  # Output: 2/1
print(float(r1))  # Output: 0.5
### 8.2  8.3 8.4 8.6 8.7  not finish yeet 


## Numpy 
import numpy as np

# Exercise 9.1: Matrix operations
n, m = 200, 500
A = np.random.randn(n, m)
B = np.random.randn(m, m)

A_plus_A = A + A

A_squared = A @ A.T
A_times_A = A * A
AB = A @ B

def compute_A_B_I(A, B, alpha):
    I = np.eye(m)
    return A @ (B - alpha * I)

# Exercise 9.2: Solving a linear system
b = np.random.randn(m)
x = np.linalg.solve(B, b)

# Exercise 9.3: Norms
frobenius_norm_A = np.linalg.norm(A, 'fro')
infinity_norm_B = np.linalg.norm(B, np.inf)
singular_values_B = np.linalg.svd(B, compute_uv=False)
largest_singular_B = max(singular_values_B)
smallest_singular_B = min(singular_values_B)

# Exercise 9.4: Power iteration
def power_iteration(Z, num_iter=1000, tol=1e-6):
    b_k = np.random.rand(Z.shape[1])
    for _ in range(num_iter):
        b_k1 = Z @ b_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        if np.linalg.norm(Z @ b_k - b_k1_norm * b_k) < tol:
            break
    return b_k1_norm, b_k

Z = np.random.randn(n, n)
largest_eigenvalue, eigenvector = power_iteration(Z)

# Exercise 9.5: Singular values
p = 0.5
C = (np.random.rand(n, n) < p).astype(int)
singular_values_C = np.linalg.svd(C, compute_uv=False)
largest_singular_C = max(singular_values_C)

# Exercise 9.6: Nearest neighbor
def nearest_neighbor(z, A):
    return A[(np.abs(A - z)).argmin()]

# Example usage
random_array = np.random.randn(100)
z_value = 0.3
nearest = nearest_neighbor(z_value, random_array)

print("Largest eigenvalue of Z:", largest_eigenvalue)
print("Largest singular value of C:", largest_singular_C)
print("Nearest neighbor to", z_value, "is", nearest)



### scipy 
from scipy.linalg import lstsq
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform

# Exercise 10.1: Least Squares
def least_squares():
    print("\nExercise 10.1: Least Squares")
    m, n = 5, 3
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    x, residuals, _, _ = lstsq(A, b)
    print("Solution x:", x)
    print("Norm of residual:", np.linalg.norm(residuals))

# Exercise 10.2: Optimization
def optimization():
    print("\nExercise 10.2: Optimization")
    def f(x):
        return - (np.sin(x - 2) ** 2) * np.exp(-x**2)
    result = minimize_scalar(f)
    print("Maximum at x:", result.x)
    print("Maximum value:", -result.fun)

# Exercise 10.3: Pairwise Distances
def pairwise_distances():
    print("\nExercise 10.3: Pairwise Distances")
    n = 5  # Number of cities
    X = np.random.rand(n, 2)
    distances = squareform(pdist(X))
    print("Pairwise distances:\n", distances)

# Run exercises
# if __name__ == "__main__":
#     least_squares()
#     optimization()
#     pairwise_distances()

### matplotlib   
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Exercise 11.1: Plotting a function
x = np.linspace(0, 2, 100)
f_x = (np.sin(x**2))**2 * np.exp(-x**2)

plt.figure(figsize=(8, 5))
plt.plot(x, f_x, label=r'$f(x) = \sin^2(x^2)e^{-x^2}$', color='b')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of f(x)')
plt.legend()
plt.grid()
plt.show()

# Exercise 11.2: Data Matrix and Estimator
X = np.random.randn(20, 10)  # 20 observations, 10 variables
b_true = np.random.randn(10)  # True parameters
z = np.random.randn(20)  # Standard normal noise
y = X @ b_true + z  # Response vector

# Estimating b using least squares: b_hat = (X^T X)^(-1) X^T y
b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

plt.figure(figsize=(8, 5))
plt.plot(b_true, 'bo-', label='True parameters')
plt.plot(b_hat, 'ro-', label='Estimated parameters')
plt.xlabel('Parameter index')
plt.ylabel('Value')
plt.title('True vs Estimated Parameters')
plt.legend()
plt.grid()
plt.show()

# Exercise 11.3: Histogram and Density Estimation
z = np.random.standard_t(df=3, size=10000)  # Example of an exotic distribution (Student's t with df=3)

plt.figure(figsize=(8, 5))
plt.hist(z, bins=25, density=True, alpha=0.6, color='g', label='Histogram')

# Gaussian Kernel Density Estimation
kde = stats.gaussian_kde(z)
x_vals = np.linspace(min(z), max(z), 1000)
plt.plot(x_vals, kde(x_vals), 'r-', label='Density Estimation')

plt.xlabel('z')
plt.ylabel('Density')
plt.title('Histogram and Density Estimation')
plt.legend()
plt.grid()
plt.show()

## Recursion 

# Exercise 12.1: Power
def power(a, b):
    if b == 0:
        return 1  # base case
    else:
        return a * power(a, b - 1)  # recursive case

# Exercise 12.2: Recursive map and filter
# Recursive map function
def myrecmap(func, lst):
    if not lst:
        return []
    return [func(lst[0])] + myrecmap(func, lst[1:])

# Recursive filter function
def myrecfilter(func, lst):
    if not lst:
        return []
    if func(lst[0]):
        return [lst[0]] + myrecfilter(func, lst[1:])
    return myrecfilter(func, lst[1:])

# Exercise 12.3: Purify
# Iterative function
def purify_iter(lst):
    return [x for x in lst if x % 2 == 0]

# Recursive function
def purify_rec(lst):
    if not lst:
        return []
    if lst[0] % 2 == 0:
        return [lst[0]] + purify_rec(lst[1:])
    return purify_rec(lst[1:])

# Exercise 12.4: Product
# Iterative function
def product_iter(lst):
    product = 1
    for num in lst:
        product *= num
    return product

# Recursive function
def product_rec(lst):
    if not lst:
        return 1
    return lst[0] * product_rec(lst[1:])

# Exercise 12.5: Factorial
def factorial(n):
    if n == 0:
        return 1  # base case
    else:
        return n * factorial(n - 1)  # recursive case

# Exercise 12.6: Recursive root finding (Newton's Method)
def find_root(f, f_prime, x, tolerance=1e-6):
    # Base case: if the function value is close enough to zero, return x
    if abs(f(x)) < tolerance:
        return x
    # Recursive case: update x using Newton's method
    return find_root(f, f_prime, x - f(x) / f_prime(x), tolerance)

# Example function
def f(x):
    return x**2 - 2  # Example function: x^2 - 2 = 0

# Derivative of f(x)
def f_prime(x):
    return 2 * x  # Derivative of x^2 - 2 is 2x

# Exercise 12.7: Collatz sequence
def collatz(n):
    if n == 1:
        return [1]
    elif n % 2 == 0:
        return [n] + collatz(n // 2)
    else:
        return [n] + collatz(3 * n + 1)

# Exercise 12.8: Fibonacci sequence
# Non-recursive Fibonacci
def fibonacci_non_rec(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Recursive Fibonacci
def fibonacci_rec(n):
    if n <= 1:
        return n
    return fibonacci_rec(n - 1) + fibonacci_rec(n - 2)

# Exercise 12.9: Palindromes
def largest_palindrome(s):
    if s == s[::-1]:
        return s
    return max(largest_palindrome(s[:-1]), largest_palindrome(s[1:]), key=len)

# Exercise 12.10: Quicksort
def quicksort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[0]
    left = [x for x in lst[1:] if x < pivot]
    right = [x for x in lst[1:] if x >= pivot]
    return quicksort(left) + [pivot] + quicksort(right)

# Test the functions with examples

# Exercise 12.1: Power
print("Power of 2^3:", power(2, 3))  # Output: 8

# Exercise 12.2: Recursive map and filter
print("Recursive map:", myrecmap(lambda x: x * 2, [1, 2, 3]))  # Output: [2, 4, 6]
print("Recursive filter:", myrecfilter(lambda x: x % 2 == 0, [1, 2, 3, 4]))  # Output: [2, 4]

# Exercise 12.3: Purify
print("Iterative Purify:", purify_iter([1, 2, 3, 4, 5]))  # Output: [2, 4]
print("Recursive Purify:", purify_rec([1, 2, 3, 4, 5]))  # Output: [2, 4]

# Exercise 12.4: Product
print("Iterative Product:", product_iter([1, 2, 3, 4]))  # Output: 24
print("Recursive Product:", product_rec([1, 2, 3, 4]))  # Output: 24

# Exercise 12.5: Factorial
print("Factorial of 5:", factorial(5))  # Output: 120

# Exercise 12.6: Recursive root finding (Newton's Method)
print("Root of x^2 - 2:", find_root(f, f_prime, 1))  # Output: Approx. 1.414 (sqrt(2))

# Exercise 12.7: Collatz sequence
print("Collatz sequence for 6:", collatz(6))  # Output: [6, 3, 10, 5, 16, 8, 4, 2, 1]

# Exercise 12.8: Fibonacci sequence
print("Non-recursive Fibonacci (F10):", fibonacci_non_rec(10))  # Output: 55
print("Recursive Fibonacci (F10):", fibonacci_rec(10))  # Output: 55

# Exercise 12.9: Palindromes
print("Largest palindrome in 'abcdba':", largest_palindrome("abcdba"))  # Output: 'abcba'

# Exercise 12.10: Quicksort
print("Sorted list:", quicksort([3, 1, 4, 1, 5, 9, 2, 6]))  # Output: [1, 1, 2, 3, 4, 5, 6, 9]



### Iterators 
import numpy as np

# Exercise 13.1: Collatz Sequence Generator
def collatz_sequence(n):
    print(f"--- Collatz Sequence for {n} ---")
    while n != 1:
        print(n)  # Print the current number
        if n % 2 == 0:
            n //= 2  # If the number is even, divide it by 2
        else:
            n = 3 * n + 1  # If the number is odd, multiply by 3 and add 1
    print(1)  # Finally print 1 when the sequence ends

# Command to run Exercise 13.1
collatz_sequence(103)

# Exercise 13.2: Collatz Sequence with NumPy
def collatz_sequence_numpy(n):
    print(f"--- Collatz Sequence for {n} (NumPy) ---")
    sequence = []  # Create an empty list to store the sequence
    while n != 1:
        sequence.append(n)  # Add the current number to the sequence
        if n % 2 == 0:
            n //= 2  # If the number is even, divide it by 2
        else:
            n = 3 * n + 1  # If the number is odd, multiply by 3 and add 1
    sequence.append(1)  # Finally, add 1 to the sequence
    return np.array(sequence)  # Return the sequence as a NumPy array

# Command to run Exercise 13.2
collatz_array = collatz_sequence_numpy(61)
print(collatz_array)

# Exercise 13.3: Prime Numbers Iterator
def is_prime(n):
    if n <= 1:
        return False  # 1 and numbers less than 1 are not prime
    for i in range(2, n):  # Check if n is divisible by any number from 2 to n-1
        if n % i == 0:
            return False  # If divisible, n is not prime
    return True  # If no divisors were found, n is prime

def prime_numbers(n):
    print(f"--- First {n} Prime Numbers ---")
    primes = []  # List to store prime numbers
    num = 2  # Start checking from 2
    while len(primes) < n:  # Keep finding primes until we have n primes
        if is_prime(num):
            primes.append(num)  # Add prime number to the list
        num += 1  # Move to the next number
    return primes

# Command to run Exercise 13.3
first_10_primes = prime_numbers(10)
print(first_10_primes)



### Exception Handling
import sys
import collections

# Exercise 14.1: Rational Numbers (Exception Handling for Denominator)
class Rational:
    def __init__(self, numerator, denominator):
        # Check if denominator is 0 and raise an exception
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

# Test for Rational class (Example of Exercise 14.1)
def test_rational():
    try:
        r = Rational(1, 0)  # This will raise an exception
    except ValueError as e:
        print(f"Exercise 14.1 Error: {e}")  # Handle the error gracefully

# Exercise 14.2: Wordcount (Counting Most Common Words in a File)
def wordcount(filename, k):
    try:
        # Open the file and read content
        with open(filename, 'r') as file:
            words = file.read().lower().split()  # Read and split words, converting to lowercase
            counter = collections.Counter(words)  # Count the frequency of each word
            common_words = counter.most_common(k)  # Get the k most common words
            
            print("\nExercise 14.2: Most Common Words")
            for word, count in common_words:
                print(f"{word}: {count}")
    except FileNotFoundError:
        print(f"Exercise 14.2 Error: The file '{filename}' was not found.")
    except ValueError:
        print("Exercise 14.2 Error: Please provide a valid integer for the number of most common words.")
    except Exception as e:
        print(f"Exercise 14.2 Error: An error occurred: {e}")

## testing 
# def main():
#     # Test Exercise 14.1
#     test_rational()

#     # Command line processing for Exercise 14.2
#     if len(sys.argv) != 3:
#         print("Usage: python exercise14.py <filename> <k>")
#     else:
#         filename = sys.argv[1]
#         try:
#             k = int(sys.argv[2])  # Convert the second argument to an integer
#             wordcount(filename, k)  # Call the wordcount function for Exercise 14.2
#         except ValueError:
#             print("Exercise 14.2 Error: The second argument must be an integer.")

# if __name__ == "__main__":
#     main()

####
### Unit testing 

import unittest
import collections

# Exercise 15.1: Factorial Function and Unit Test
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Exercise 15.2: Prime Numbers Function and Unit Test
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def primes_between(a, b):
    if a > b:
        raise ValueError("a must be less than or equal to b.")
    return [n for n in range(a, b + 1) if is_prime(n)]

# Exercise 15.3: Quicksort Function and Unit Test
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


# Unit Tests for all Exercises
class TestAllFunctions(unittest.TestCase):

    # Test for Exercise 15.1: Factorial Function
    def test_factorial_positive(self):
        self.assertEqual(factorial(5), 120)  # 5! = 120
        self.assertEqual(factorial(3), 6)    # 3! = 6

    def test_factorial_zero(self):
        self.assertEqual(factorial(0), 1)    # 0! = 1

    def test_factorial_negative(self):
        with self.assertRaises(ValueError):  # Test for negative number
            factorial(-5)

    # Test for Exercise 15.2: Prime Numbers
    def test_is_prime(self):
        self.assertTrue(is_prime(5))   # 5 is prime
        self.assertFalse(is_prime(4))  # 4 is not prime
        self.assertFalse(is_prime(1))  # 1 is not prime

    def test_primes_between(self):
        self.assertEqual(primes_between(10, 20), [11, 13, 17, 19])  # Primes between 10 and 20
        self.assertEqual(primes_between(1, 10), [2, 3, 5, 7])        # Primes between 1 and 10

    def test_primes_between_invalid(self):
        with self.assertRaises(ValueError):  # Test when a > b
            primes_between(20, 10)

    # Test for Exercise 15.3: Quicksort Function
    def test_quicksort(self):
        self.assertEqual(quicksort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]), [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9])
        self.assertEqual(quicksort([2, 2, 2, 2]), [2, 2, 2, 2])  # All elements are the same
        self.assertEqual(quicksort([10, -5, 0, 7]), [-5, 0, 7, 10])  # Negative numbers included

    def test_empty_list(self):
        self.assertEqual(quicksort([]), [])  # Empty list should return empty list


# Running all the tests
# if __name__ == '__main__':
#     unittest.main()

### More modules 
## install pip3 install requests 
## install pip3 install flasks 

import requests
import re
from flask import Flask
from datetime import datetime

# Exercise 16.1: Regular Expressions to Find Email Addresses

def find_email_addresses():
    # Step 1: Download the data using requests
    url = "http://stanford.edu/~schmit/cme193/ex/data/emailchallenge.txt"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Step 2: Use regex to find email addresses
        text = response.text
        email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

        emails = re.findall(email_pattern, text)

        # Step 3: Print out the local and domain parts separately
        for local, domain in emails:
            print(f"Local part: {local}, Domain part: {domain}")
    else:
        print("Failed to retrieve the data")


# Exercise 16.2: Flask App to Display Current Date and Time

app = Flask(__name__)

@app.route('/')
def show_datetime():
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"Current date and time: {current_time}"


# Main Function to Call the Exercises
if __name__ == '__main__':
    # Exercise 16.1: Find Email Addresses
    print("Finding Email Addresses from the URL...\n")
    find_email_addresses()

    # Exercise 16.2: Run Flask App
    print("\nRunning Flask App for Current Date and Time...")
    app.run(debug=True)
