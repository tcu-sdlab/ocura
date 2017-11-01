import struct
from PIL import Image, ImageEnhance

filename = 'ETL6/ETL6C_09'
id_record = 13830  #character number
sz_record = 2052  #size of 1 record

for i in range(0, id_record, 1):

    with open(filename, 'r') as f:
        #f.seek(id_record * sz_record)
        f.seek(i * sz_record)
        s = f.read(sz_record)
        r = struct.unpack('>2x1s1x28x2016s4x', s)
        iF = Image.frombytes('F', (64, 63), r[1], 'bit', 4)
        iP = iF.convert('P')
        #fn = 'ETL3C_{0:d}_{1:d}.png'.format(i/100,i%100)  #1
        fn = 'ETL6C_a{0}_{1:d}.png'.format(r[0],(i%1383))  #2
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(fn, 'PNG')