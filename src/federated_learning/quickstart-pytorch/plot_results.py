import matplotlib.pyplot as plt

rounds = list(range(1, 21))

accuracy = [
0.4424,0.5759,0.6315,0.6664,0.6855,
0.6964,0.7093,0.7191,0.7243,0.7360,
0.7405,0.7425,0.7456,0.7489,0.7593,
0.7548,0.7571,0.7585,0.7553,0.7609
]

loss = [
1.8120,1.1984,1.0447,0.9559,0.9124,
0.8748,0.8507,0.8351,0.8059,0.7895,
0.7831,0.7716,0.7836,0.7823,0.7570,
0.7749,0.7812,0.7760,0.7850,0.7837
]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(rounds, accuracy)
plt.title("Accuracy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")

plt.subplot(1,2,2)
plt.plot(rounds, loss)
plt.title("Loss vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()