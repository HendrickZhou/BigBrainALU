def estimate_cap(layers, output_dims):
    cap = 0
    for i, units in enumerate(layers):
        if i is 0:
            cap = cap + 3 * units
        else:
            cap = cap + min(units*3, last_unit)
        last_unit = units
    cap = cap + min(output_dims, 3*last_unit)
    return cap


layers = [45, 40, 35, 30, 25, 10, 15]
print(estimate_cap(layers, 1))