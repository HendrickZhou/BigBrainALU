class ALU:
    """
    Op_code:
    ********
    bit logical
    ********
    a : and
    o : or
    x : xor
   
    1ca : 1's complement of A
    1cb : 1's complemnt of B
    
    ********
    arithmetic
    ********
    ad : add
    ac : add with carry
    su : sub
    sub : sub with borrow
    # msl : signed multiple lower
    # msh : signed multiple higher
    mul : unsigned multiple lower
    muh : unsigned mulitple higher
    d : divide
    r : remu

    2ca : 2's complement of A
    2cb : 2's complement of B

    ia : increment A
    ib : increment B
    da : decrement A
    da : decrement B
    pa : pass A
    pb : pass B
    
    ********
    bit shift
    ********
    asl : arithmetic shift left
    asr : arithmetic shift right
    lsl : logical shift left
    lsr : logical shift right

    """

    def __init__(self, num_bit = 8, op_code_list = ['ad', 'ac', 'su', 'mul', 'muh', 'd', 'r', 'a', 'o', 'x', 'lsl','lsr', 'ror', 'rol']):
        self.op_dict = {
            'ad' : self.ADD,
            'ac' : self.ADDC,
            'su' : self.SUB,
            'mul' : self.MUL,
            'muh' : self.MUH,
            'd' : self.DIV,
            'r' : self.REMU,
            # 'sub' : self.SUBB,
            'a' : self.AND,
            'o' : self.OR,
            'x' : self.XOR,

            'lsl' : self.LSL,
            'lsr' : self.LSR,

            'ror' : self.ROR,
            'rol' : self.ROL,
        }
        self.bits = num_bit
        self.op_code_list = op_code_list
        self.op_code_len = op_code_list.__len__().bit_length()
        op_bits = []
        meta_dic = dict()
        for i in range(len(op_code_list)):
            meta_dic[op_code_list[i]] = self.toNBit(i+1, self.op_code_len)
        self.meta_dic = meta_dic
        self.data_dim = 2*self.bits + self.op_code_len
        self.label_dim = self.bits

    def __call__(self, A, B, op_code):
        """
        Plz use positive integer as input, as we only handle unsigned numbers
        """
        if not self._valid_input(A) or not self._valid_input(B):
            raise Exception("input out of range")
        if op_code not in self.op_dict:
            raise Exception("Illegal op_code")
        return self.toNBit(A), self.toNBit(B), self.meta_dic[op_code], self.op_dict[op_code](A, B)

    def _valid_input(self, inputd):
        if inputd.bit_length() > self.bits or inputd < 0:
            return False
        return True

    def AND(self, A, B):
        return self.toNBit(A & B)

    def OR(self,A,B):
        return self.toNBit(A | B)

    def XOR(self,A, B):
        return self.toNBit(A ^ B)

    def ADD(self,A, B):
        return self.toNBit(A + B)

    def ADDC(self,A, B):
        return self.toNBit(A+B+1)

    def SUB(self,A,B):
        return self.toNBit(A-B)

    def LSL(self, A, B):
        return self.toNBit(A<<B)

    def LSR(self, A, B):
        return self.toNBit(A>>B)

    def ROR(self, A, B):
        A_string = self.toNBit(A)
        B_rem = B % self.bits
        return A_string[-B_rem:] + A_string[:-B_rem]

    def ROL(self, A, B):
        A_string = self.toNBit(A)
        B_rem = B % self.bits
        return A_string[B_rem:] + A_string[:B_rem]

    def MUL(self, A, B):
        return self.toNBit(A*B, bits = self.bits * 2)[-self.bits:]

    def MUH(self, A, B):
        return self.toNBit(A*B, bits = self.bits * 2)[:self.bits]

    def DIV(self, A, B):
        """
        if 0 return A untouched
        """
        if B is 0:
            return self.toNBit(A)
        return self.toNBit(A//B)

    def REMU(self, A, B):
        """
        if 0 return 0
        """
        if B is 0:
            return self.toNBit(0)
        return self.toNBit(A%B)


    # def SUBB(self,A, B):
    #     return self.toNBit(A-B)

    def gen_range(self):
        # give the bit, generate the decimal range for us
        self.low = 0
        self.high = int('1' * self.bits, 2)
        return range(self.low, self.high + 1), self.op_code_list
    
    def toNBit(self, number, bits=None):
        if not bits:
            if number > 0:
                b =  bin(number)[2:][-self.bits:]
                if number.bit_length() < self.bits:
                    return (self.bits - number.bit_length())*'0' + b
                else:
                    return b
            else:
                min_bits = max(self.bits, number.bit_length())
                pos = number + (1 << min_bits)
                if pos.bit_length() < self.bits:
                    b = (self.bits - pos.bit_length())*'0' + bin(pos)[2:]
                    return b
                return bin(pos)[2:][-self.bits:]       


        else:
            if number > 0:
                b =  bin(number)[2:][-bits:]
                if number.bit_length() < bits:
                    return (bits - number.bit_length())*'0' + b
                else:
                    return b
            else:
                min_bits = max(bits, number.bit_length())
                pos = number + (1 << min_bits)
                if pos.bit_length() < bits:
                    b = (bits - pos.bit_length())*'0' + bin(pos)[2:]
                    return b
                return bin(pos)[2:][-bits:]       


#def toNBit(number, bits):
#    if number > 0:
#        b =  bin(number)[2:][-bits:]
#        if number.bit_length() < bits:
#            return (bits - number.bit_length())*'0' + b
#        else:
#            return b
#    else:
#        min_bits = max(bits, number.bit_length())
#        pos = number + (1 << min_bits)
#        if pos.bit_length() < bits:
#            b = (bits - pos.bit_length())*'0' + bin(pos)[2:]
#            return b
#        return bin(pos)[2:][-bits:]       

if __name__ == "__main__":
    dumbALU = ALU()
    # print(dumbALU.meta_dic)

    # for n in dumbALU.gen_range():
    #     for m in dumbALU.gen_range():
    #         print(dumbALU(n, m, 'a'))

    # print(dumbALU(10, 0, 'ror'))
    # for i in range(16):
    #     print(dumbALU(1, i, 'ror'))

    # print(dumbALU(10, 1, 'mul'))
    # print(dumbALU(10, 89, 'muh'))
    # print(dumbALU(10, 0, 'd'))
    # print(dumbALU(11, 0, 'r'))


