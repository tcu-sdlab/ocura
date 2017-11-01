in_file = open('ETL3/ETL3C_2','rb')  #change ETL3C_1 to ETL3C_2 in need
out_file = open('ETL3/ETL3C_2_converted','wb+')

bits = [b'\x00',b'\x30',b'\x3c',b'\x3f']

read_data = [b'\x00']

while True:


    if not read_data:
        break

    for num in range(72):  #72*3=216byte
        data = [b'\x00']


        for j in range(3):  #read 3bytes
            read_data = in_file.read(1)
            if not read_data:
                break

            data.append(read_data)

        if not read_data:
            break

        data.append(b'\x00')

        for i in range(4):
            a = int.from_bytes(data[i],'little')
            b = int.from_bytes(data[i+1],'little')

            d = ( (a<<(6-(i*2))) & int.from_bytes(bits[i],'little') ) | (b>>(i*2+2)) 

            out_file.write(d.to_bytes(1,'little'))

    if not read_data:
        break

    read_data = in_file.read(2736)
    out_file.write(read_data)


in_file.close()
out_file.close()
