from sage.crypto.boolean_function import BooleanFunction


def is_linear_sbox(sbox: list[int]) -> bool:
    """
    Determine if an S-Box is linear
    """
    input_size = (len(sbox) - 1).bit_length()
    output_size = max(sbox).bit_length()

    print(f"[*] Input bits: {input_size}, Output bits: {output_size}")

    is_linear = True

    for bit_index in range(output_size):
        truth_table = [(sbox[i] >> bit_index) & 1 for i in range(2**input_size)]
        f = BooleanFunction(truth_table)

        deg = f.algebraic_degree()
        print(f"[+] The {bit_index} bit output function's algebra degree: {deg}")

        if deg > 1:
            is_linear = False
            break

    return is_linear