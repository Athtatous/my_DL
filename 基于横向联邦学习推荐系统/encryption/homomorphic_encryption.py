import tenseal as ts

# 创建一个加密上下文
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,  # 加密方案类型，这里使用CKKS
        poly_modulus_degree=8192,  # 多项式模数的次数
        coeff_mod_bit_sizes=[60, 40, 40, 60]  # 系数模数的比特数列表
    )
    context.global_scale = 2 ** 40  # 全局缩放因子
    context.generate_galois_keys()  # 生成伽罗瓦密钥
    return context

# 对数据进行加密
def encrypt_data(context, data):
    encrypted_data = [ts.ckks_vector(context, vector) for vector in data]  # 使用CKKS向量对数据进行加密
    return encrypted_data

# 对加密数据进行解密
def decrypt_data(context, encrypted_data):
    decrypted_data = [vector.decrypt() for vector in encrypted_data]  # 对每个向量进行解密
    return decrypted_data