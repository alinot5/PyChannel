import cupy as cp

x = cp.random.rand(5000, 5000)
y = cp.fft.fft2(x)

print("Done")
input("Press Enter to exit...")