import math

# Ena 2D točka v latentnem prostoru
z1 = 1.0
z2 = 2.0

# Preprosti funkciji s in t
def s(z1):
    return 0.3 * z1

def t(z1):
    return 0.5 * z1


# FORWARD transformacija Real NVP coupling layerja
x1 = z1
x2 = z2 * math.exp(s(z1)) + t(z1)

print("Originalna točka z:")
print((z1, z2))

print("\nPo forward transformaciji:")
print((x1, x2))

# Log-determinanta Jacobiana
log_det = s(z1)

print("\nlog|det(J)|:")
print(log_det)


# INVERSE transformacija
recovered_z1 = x1
recovered_z2 = (x2 - t(x1)) * math.exp(-s(x1))

print("\nRekonstruirana točka z:")
print((recovered_z1, recovered_z2))