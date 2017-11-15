def main(in_file, out_file):
    masks = list(b'\x00\x30\x3c\x3f')

    try:
        for num in range(73):  #72*3=216byte
            data = [0]
            data.extend(in_file.read(3))
            data.append(0)

            for lhs, mask, i, rhs in zip(data, masks, range(0, 8, 2), data[1:]):
                # ((lhs << 6 - (i * 2 )) & mask | rhs >> (i * 2 + 2)
                d = ((lhs<<(6 - i)) & mask) | (rhs>>(i + 2))

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
