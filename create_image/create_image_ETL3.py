import struct
from PIL import Image, ImageEnhance

filename = 'ETL3/ETL3C_1_converted'  #change ETL3C_1_converted to ETL3C_2_converted in need
id_record = 4800  #character number
sz_record = 3024  #size of 1 record3936

for i in range(0, id_record, 1):

    with open(filename, 'r') as f:
        #f.seek(id_record * sz_record)
        f.seek(i * sz_record)
        s = f.read(sz_record)
        # r = struct.unpack('>24x4s260x3648s', s)
        r = struct.unpack('>24x4s260x2736s', s)
        #print (r[0], r[1])
        iF = Image.frombytes('F', (72, 76), r[1], 'bit', 4)
        iP = iF.convert('P')
        #fn = 'ETL3C_{0:d}_{1:d}.png'.format(i/100,i%100)  #1
        fn = 'ETL3C_{0:d}_{1:d}.png'.format(i/100,(i%100)+100)  #2
        enhancer = ImageEnhance.Brightness(iP)
        iE = enhancer.enhance(16)
        iE.save(fn, 'PNG')
