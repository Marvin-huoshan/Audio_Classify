import matplotlib.pyplot as plt

# Data for the number of samples
categories = ['Cicada', 'Termite', 'Cricket', 'Beetle']
values = [433, 399, 1092, 2764]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['blue', 'green', 'gray', 'gray'])

plt.title("Number of Instance for Each Insect Class")
plt.xlabel("Insect Class")
plt.ylabel("Number of Instance")


plt.savefig('Instance_number.pdf')