import tenseal as ts


def gen_ckks_keys():
    print("====== CKKS =======")

    # coefficient modulus size
    # sum<=
    # 8192:218
    # 16384:438
    # 32768:881
    # 8192: [60, 40, 40, 60]
    # 8192 [50,30,30,30,50]
    # 8192 [30, 25, 25, 25, 25, 25, 25, 30]
    # 8192: [40, 21, 21, 21 , 21, 21, 21, 40]
    # 16384: [60, 40, 40, 40, 40, 40, 40, 40, 60]
    # 32768: [60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    # context = ts.context(
    #     ts.SCHEME_TYPE.CKKS,
    #     poly_modulus_degree=16384,
    #     coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]
    # )
    # context = ts.context(
    #     ts.SCHEME_TYPE.CKKS,
    #     poly_modulus_degree=32768,
    #     coeff_mod_bit_sizes=[60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60]
    # )

    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    ctx_file = "./ts_ckks.config"
    context_bytes = context.serialize(save_secret_key=True)
    f = open(ctx_file, "wb")
    f.write(context_bytes)
    f.close()

    pk_file = "./ts_ckks_pk.config"
    pk_bytes = context.serialize(save_secret_key=False)
    f = open(pk_file, "wb")
    f.write(pk_bytes)
    f.close()


if __name__ == '__main__':
    gen_ckks_keys()
