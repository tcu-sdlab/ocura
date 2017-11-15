def main(in_file, out_file):
    bits = [b'\x00', b'\x30', b'\x3c', b'\x3f']
    bits = [int.from_bytes(i, 'little') for i in bits]

    try:
        for num in range(73):  #72*3=216byte
            data = [0]
            data.extend(in_file.read(3))
            data.append(0)

            for lhs, mask, rhs in zip(data, bits, data[1:]):
                d = ((lhs<<(6-(i*2))) & mask) | (rhs>>(i*2+2))

                out_file.write(d.to_bytes(1, 'little'))

        read_data = in_file.read(2736)
        out_file.write(read_data)
    except IOError:
        print("cannot read data")
        raise

if __name__ == '__main__':
    in_file = open('ETL3/ETL3C_2', 'rb')  #change ETL3C_1 to ETL3C_2 in need
    out_file = open('ETL3/ETL3C_2_converted', 'wb+')

    main(in_file, out_file)

    in_file.close()
    out_file.close()
