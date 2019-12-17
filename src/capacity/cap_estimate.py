def cap_estimate(layers, input_dims,  output_dims):
    cap = 0
    for i, units in enumerate(layers):
        if i is 0:
            cap = cap + (input_dims+1) * units
        else:
            cap = cap + min((last_unit+1)*units, last_unit)
        last_unit = units
    cap = cap + min(output_dims, output_dims*(last_unit+1))
    return cap

def kill_cap_by(layers, cap):
   pass 

if __name__ == "__main__":
    layers = [12, 11, 10, 9, 8]
    print(cap_estimate(layers, 20, 1))
