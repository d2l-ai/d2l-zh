t = True
f = False
print(type(f))

print(t and f)


hello = 'hello'
world = 'world'
hw12 = '%s %s %d' % (hello, world, 12)
print(hw12)

s = 'hello'
print(s.capitalize())
print(s.upper())
print(s.rjust(7))
print(s.center(7))
print(s.replace('l', '(ell)'))
print('  world  '.strip())

print(
"""
Containers
""")

xs = [3, 1, 2]
print(xs[-1])
xs[2] = 'foo'
xs.append('bar')
print(xs)
x = xs.pop()
print(x, xs)


nums = list(range(5))
print(nums)
print(nums[:])
print(nums[:-2])
nums[2:4] = [8,9]
print(nums)


print('\n Loops\n')

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx+1, animal))

nums = [0, 1,2,3,4]
squares = []
for x in nums:
    squares.append(x ** 2)

print(squares)


print('\nList Comprehension\n')
squares = [x ** 2 for x in nums]
print(squares)

even_squares = [x**2 for x in nums if x % 2 == 0]
print(even_squares)


d = {'cat': 'cute', 'dog': 'furry'}
print('cat' in d)
print(d.get('moneky', 'N/A'))
print(d.get('moneky'))
d['fish'] = 'wet'
print(d)
del d['fish']
print(d.get('fish', 'N/A'))


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))


print('\nDic comprehensions\n')
nums = list(range(5))
even_num_to_square = {x: x**2 for x in nums if x % 2 == 0}
print(even_num_to_square)

animals = {'cat', 'dog'}
print('cat' in animals)
print('fish' in animals)
animals.add('fish')
print('fish' in animals)
animals.remove('cat')
print(len(animals))


animals = {'cat', 'dog', 'fish'}
for animal in animals:
    print(animal)


for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx+1, animal))

print('\nSet comprehension\n')
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)

s  = {(2,2)}
print(s)
d = {(x,x+1):x for x in range(10)}
print(d)
t = (5,6)
print(type(t))
print(d[t])
print(d[1,2])

class Greeter(object):
    def __init__(self, name):
        self.name = name

    def greet(self, loud=False):
        if loud:
            print('HELLO, %s' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')
g.greet()
g.greet(True)



def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3, 6, 8, 10, 1, 2, 1]))